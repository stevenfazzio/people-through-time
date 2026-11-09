"""Stage 13: screenshot the page -> docs/social-preview.png, the 1200 x 630 link card.

Decided 2026-11-09 (CLAUDE.md, "Publishing"): the card is the interactive page itself, as on the sibling maps,
so a shared link previews what the reader gets. The page is WebGL, so this drives the installed Chrome through
Playwright (`uv run --extra publish`; no bundled browser is downloaded), headless, at 2x, and downsamples. The
page fetches its data from files beside it, so docs/ is served over http on a free local port.

The slider is set to config.SOCIAL_PREVIEW_WINDOW and the view to the page's fit plus config.SOCIAL_PREVIEW_ZOOM
through the page's debug handle (window.__ptt), so the card is reproducible. The title panel is hidden: link
previews print og:title under the image. The timeline stays.

Input: docs/ (stage 12's data and the page). Output: docs/social-preview.png.
"""

import functools
import http.server
import io
import threading
import time

from PIL import Image
from playwright.sync_api import sync_playwright

import config
from util import log, write_bytes_atomic

SIZE = (1200, 630)
SCALE = 2
SETTLE_MS = 5000


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main() -> None:
    t0 = time.monotonic()
    handler = functools.partial(QuietHandler, directory=str(config.FRONTEND_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": SIZE[0], "height": SIZE[1]}, device_scale_factor=SCALE)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url, wait_until="load")
            page.wait_for_function("window.__ptt && window.__ptt.S.fitted", timeout=60_000)
            page.mouse.move(0, 0)  # no hovercard in the shot
            page.evaluate(
                """([w, dz]) => {
                  const P = window.__ptt;
                  P.setS(w);
                  P.deck().setProps({ initialViewState: { target: P.S.fitTarget, zoom: P.S.zoom0 + dz,
                    minZoom: P.S.zoom0 - 1.5, maxZoom: P.S.zoom0 + 7 } });
                  document.getElementById('title').style.display = 'none';
                }""",
                [config.SOCIAL_PREVIEW_WINDOW, config.SOCIAL_PREVIEW_ZOOM],
            )
            page.wait_for_timeout(SETTLE_MS)
            shot = page.screenshot()
            browser.close()
        if errors:
            raise SystemExit(f"page errors: {errors}")
    finally:
        server.shutdown()
    img = Image.open(io.BytesIO(shot)).convert("RGB").resize(SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    write_bytes_atomic(buf.getvalue(), config.SOCIAL_PREVIEW_PNG)
    log(f"Wrote {config.SOCIAL_PREVIEW_PNG.relative_to(config.REPO_ROOT)} ({len(buf.getvalue()) / 1e3:.0f} KB) "
        f"of window {config.SOCIAL_PREVIEW_WINDOW} at zoom +{config.SOCIAL_PREVIEW_ZOOM} in "
        f"{time.monotonic() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
