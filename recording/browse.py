"""Record smooth scroll-throughs of web pages (Playwright video + visible cursor).
Usage: python browse.py <name> <url> [seconds] [scroll_px]"""
import math, os, random, shutil, sys, time, glob, subprocess
from playwright.sync_api import sync_playwright

W, H = 1600, 900
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "footage")
CURSOR_JS = """(() => { const c=document.createElement('div');
c.style.cssText='position:fixed;z-index:2147483647;width:24px;height:24px;pointer-events:none;left:800px;top:450px';
c.innerHTML='<svg width="24" height="24" viewBox="0 0 24 24"><path d="M4 2 L20 12 L12.5 13.5 L9 21 Z" fill="#fff" stroke="#000" stroke-width="1.4"/></svg>';
const add=()=>document.body?document.body.appendChild(c):requestAnimationFrame(add);add();
document.addEventListener('mousemove',e=>{c.style.left=e.clientX+'px';c.style.top=e.clientY+'px'},true);})();"""

name, url = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 25
scroll_px = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
tmp = os.path.join(OUT, "_pw_tmp")
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=tmp,
                        record_video_size={"width": W, "height": H}, color_scheme="dark")
    page = ctx.new_page()
    page.add_init_script(CURSOR_JS)
    page.goto(url, wait_until="domcontentloaded")
    time.sleep(3)
    page.mouse.move(700, 400)
    steps = int(secs * 20)
    for i in range(steps):
        t = i / steps
        page.mouse.move(700 + 250 * math.sin(t * 5) + random.uniform(-2, 2), 420 + 120 * math.sin(t * 3))
        page.mouse.wheel(0, scroll_px / steps * (0.5 + math.sin(math.pi * t)))
        time.sleep(0.05)
    time.sleep(1.5)
    vid = page.video.path()
    ctx.close(); b.close()
dst = os.path.join(OUT, f"web_{name}.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-vf", "fps=30,format=yuv420p",
                "-c:v", "libx264", "-crf", "18", "-g", "30", "-keyint_min", "30", "-movflags", "+faststart", dst], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print(dst)
