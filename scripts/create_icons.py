"""
Generate placeholder app icons for Tauri.
Run once from the project root: python scripts/create_icons.py
For a polished product icon, run: cargo tauri icon path/to/your-icon.png
"""
import os
import struct
import zlib
from pathlib import Path

ICONS_DIR = Path(__file__).parent.parent / "src-tauri" / "icons"
ICONS_DIR.mkdir(exist_ok=True)

# Blue accent color #2563EB = rgb(37, 99, 235)
R, G, B = 37, 99, 235


def make_png(w: int, h: int, r: int, g: int, b: int) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + bytes([r, g, b] * w) for _ in range(h))
    idat = chunk(b"IDAT", zlib.compress(raw, 9))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def make_ico(png_32: bytes) -> bytes:
    # ICO with one 32x32 PNG entry
    header = struct.pack("<HHH", 0, 1, 1)  # reserved, type=1 (ICO), count=1
    offset = 6 + 16  # header + one directory entry
    dir_entry = struct.pack(
        "<BBBBHHII",
        32, 32,  # width, height
        0, 0,    # color count, reserved
        1,       # planes
        32,      # bit count
        len(png_32),
        offset,
    )
    return header + dir_entry + png_32


def write(path: Path, data: bytes):
    path.write_bytes(data)
    print(f"  wrote {path.name} ({len(data)} bytes)")


print("Generating placeholder icons…")
png_32 = make_png(32, 32, R, G, B)
png_128 = make_png(128, 128, R, G, B)

write(ICONS_DIR / "32x32.png", png_32)
write(ICONS_DIR / "128x128.png", png_128)
write(ICONS_DIR / "icon.ico", make_ico(png_32))

# macOS .icns: minimal stub that Apple tooling accepts (empty ICNS container)
# For a real product, use `cargo tauri icon` or Image2icon on macOS.
icns_header = b"icns"
icns_body = b""  # no sub-images — Tauri/macOS will use a fallback
icns_size = struct.pack(">I", 8 + len(icns_body))
write(ICONS_DIR / "icon.icns", icns_header + icns_size + icns_body)

print("\nDone. For production-quality icons, run:")
print("  cargo tauri icon assets/icon.png")
print("and provide a 1024x1024 PNG as input.")
