# Examples

## `demo_ring.stl`

A four-prong solitaire ring built from primitives by [`make_demo_ring.py`](make_demo_ring.py). It is
original, rights-free and watertight (about 11 000 triangles, 22.6 x 9.2 x 28.2 mm). Like a casting model
it has **no stone**, only the conical seat the stone would sit in.

Things to try:

1. **Render** tab > *Dosyadan aç...* > `examples/demo_ring.stl`: two live views of the ring.
2. Tick **Taşları yuvalara yerleştir** ("Place stones in the seats"): the seat is detected (7.2 mm) and
   a brilliant is placed in it.
3. Change **Maden** (metal) and **Taş** (stone) colours; orbit either view; *Bu açıyı kaydet*.
4. **Analyze** tab > *Analyze file...* > dimensions, ring bore and metal weights (no AI needed).

Regenerate it (or change the seat) with:

```powershell
python examples/make_demo_ring.py
```
