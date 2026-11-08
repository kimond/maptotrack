"""Offline test of the aerial-imagery path with a fake WMS server (Saguenay coordinates)."""
import io, os, sys
os.environ["MAPTOTRACK_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from app import imagery, pipeline, routing

calls = []
def fake_get(url, params=None):
    calls.append(params)
    w, h = int(params["WIDTH"]), int(params["HEIGHT"])
    img = Image.new("RGB", (w, h), (90, 110, 70))
    d = ImageDraw.Draw(img)
    for i in range(0, w, 40):
        d.line([(i, 0), (i, h)], fill=(140, 130, 110), width=6)
    b = io.BytesIO(); img.save(b, "JPEG"); return b.getvalue()
imagery._get = fake_get

wps = [[48.49847, -71.25462], [48.49794, -71.25224], [48.49647, -71.25231], [48.49690, -71.25560]]
r = routing.route(wps, ["free"] * 4, loop=True)
rep = pipeline.generate({"name": "Imagery mock", "coords": r["coords"], "loop": True, "margin": 300}, log=print)
print("wms calls", len(calls), "first", {k: calls[0][k] for k in ("BBOX", "WIDTH", "HEIGHT", "CRS")})
print("trees", rep["trees"], "credits", rep["credits"], "warnings", rep["warnings"])
