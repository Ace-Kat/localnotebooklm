use std::process::Child;
use std::sync::Mutex;
use tauri::Manager;

pub struct BackendProcess(pub Mutex<Option<Child>>);

fn find_backend_dir() -> Option<std::path::PathBuf> {
    // At compile time, CARGO_MANIFEST_DIR = <project>/src-tauri
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir.parent()?;
    let candidate = project_root.join("backend");
    if candidate.join("main.py").exists() {
        Some(candidate)
    } else {
        None
    }
}

fn find_python(backend_dir: &std::path::Path) -> std::path::PathBuf {
    // Prefer venv python, fall back to system python
    let venv_win = backend_dir.join(".venv/Scripts/python.exe");
    let venv_unix = backend_dir.join(".venv/bin/python3");
    if venv_win.exists() {
        venv_win
    } else if venv_unix.exists() {
        venv_unix
    } else if cfg!(target_os = "windows") {
        std::path::PathBuf::from("python")
    } else {
        std::path::PathBuf::from("python3")
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let app_data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&app_data_dir)?;
            let app_data_str = app_data_dir.to_string_lossy().to_string();

            // In release builds, try the sidecar bundled by Tauri first.
            // In debug builds (cargo tauri dev), fall back to spawning Python directly.
            let child: Option<Child> = {
                let mut spawned: Option<Child> = None;

                #[cfg(not(debug_assertions))]
                {
                    // Production: find the PyInstaller-frozen backend in the app resource dir.
                    // resource_dir() returns the correct platform path:
                    //   macOS  → <App>.app/Contents/Resources/
                    //   Windows → <install>/resources/
                    if let Ok(resource_dir) = app.path().resource_dir() {
                        let bin = if cfg!(target_os = "windows") {
                            resource_dir.join("backend.exe")
                        } else {
                            resource_dir.join("backend")
                        };
                        if bin.exists() {
                            spawned = std::process::Command::new(&bin)
                                .env("APP_DATA_DIR", &app_data_str)
                                .env("CHROMA_ANONYMIZED_TELEMETRY", "False")
                                .spawn()
                                .ok();
                        }
                    }
                }

                // Dev / fallback: spawn Python from backend/ directory
                if spawned.is_none() {
                    if let Some(backend_dir) = find_backend_dir() {
                        let python = find_python(&backend_dir);
                        spawned = std::process::Command::new(&python)
                            .args(["-m", "uvicorn", "main:app",
                                   "--port", "8765", "--host", "127.0.0.1"])
                            .current_dir(&backend_dir)
                            .env("APP_DATA_DIR", &app_data_str)
                            .env("CHROMA_ANONYMIZED_TELEMETRY", "False")
                            .spawn()
                            .ok();
                    }
                }

                spawned
            };

            *app.state::<BackendProcess>().0.lock().unwrap() = child;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<BackendProcess>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
