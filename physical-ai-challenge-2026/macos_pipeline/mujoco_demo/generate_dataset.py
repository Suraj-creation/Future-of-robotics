#!/usr/bin/env python3
"""Generate synthetic RGB dataset with YOLO and COCO labels from MuJoCo renders.

This script creates a temporary MJCF that includes the SO101 XML and a
single red cube placed at random poses, renders RGB frames offscreen, and
exports YOLO-format `.txt` labels and a COCO JSON (`coco_annotations.json`).

Usage:
  python generate_dataset.py --num 200 --out ../datasets/synthetic_cube

Notes:
- Requires the `mujoco` Python package and `Pillow`/`numpy`.
- This uses a simple red-channel threshold to find the cube in the render and
  compute a 2D bounding box. It is intentionally simple and fast for bootstrapping
  YOLO training on cube-only data.
"""
import argparse
import json
import random
import tempfile
from pathlib import Path
import sys

try:
    import mujoco
except Exception as e:
    print('mujoco not available:', e)
    sys.exit(1)

from PIL import Image
import numpy as np


HERE = Path(__file__).resolve().parent
SO101_XML = HERE.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'so101_new_calib.xml'


def detect_bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return xmin, ymin, xmax, ymax


def write_coco(outdir, images_meta, ann_meta):
    coco = {
        'images': images_meta,
        'annotations': ann_meta,
        'categories': [{'id': 1, 'name': 'cube'}]
    }
    (outdir / 'coco_annotations.json').write_text(json.dumps(coco, indent=2))


def make_xml_with_cube(x, y, z, sx=0.02, sy=0.02, sz=0.02, rgba=(1, 0, 0, 1)):
    # Use absolute include path so temporary file can be loaded independently
    include = str(SO101_XML)
    cube_name = 'cube_gt'
    geom_size = f"{sx} {sy} {sz}"
    rgba_s = ' '.join(str(v) for v in rgba)
    xml = f'''<mujoco model="dataset">
  <include file="{include}" />
  <worldbody>
    <body name="{cube_name}" pos="{x} {y} {z}">
      <geom name="{cube_name}_geom" type="box" size="{geom_size}" rgba="{rgba_s}" />
    </body>
  </worldbody>
</mujoco>'''
    return xml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--out', type=Path, default=HERE.parent / 'datasets' / 'synthetic_cube')
    parser.add_argument('--w', type=int, default=640)
    parser.add_argument('--h', type=int, default=480)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    outdir = args.out
    images_dir = outdir / 'images'
    labels_dir = outdir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_meta = []
    ann_meta = []
    image_id = 1
    ann_id = 1

    for i in range(args.num):
        # sample a cube pose in front of the robot workspace (tune as needed)
        x = random.uniform(0.15, 0.35)
        y = random.uniform(-0.20, 0.20)
        z = 0.02  # place on table

        xml = make_xml_with_cube(x, y, z)

        # load model (try from string, fallback to temp file)
        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception:
            with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
                f.write(xml)
                tmp_path = Path(f.name)
            model = mujoco.MjModel.from_xml_path(str(tmp_path))

        sim = mujoco.MjData(model)

        # render
        try:
            Renderer = getattr(mujoco, 'Renderer')
            renderer = Renderer(model)
            img = renderer.render(args.w, args.h)
        except Exception as e:
            print('Renderer failed for sample', i, e)
            continue

        # save image
        img_arr = np.asarray(img)
        img_name = f'frame_{i:06d}.png'
        img_path = images_dir / img_name
        Image.fromarray(img_arr).save(img_path)

        # simple red-channel mask to find the cube
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        mask = (r > 150) & (g < 120) & (b < 120)

        bbox = detect_bbox_from_mask(mask)
        label_path = labels_dir / f'frame_{i:06d}.txt'
        if bbox is None:
            label_path.write_text('')
            print('No bbox detected for', img_name)
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            cx = (xmin + xmax + 1) / 2.0 / args.w
            cy = (ymin + ymax + 1) / 2.0 / args.h
            wn = w / args.w
            hn = h / args.h
            label_path.write_text(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

            images_meta.append({'id': image_id, 'file_name': img_name, 'width': args.w, 'height': args.h})
            ann_meta.append({'id': ann_id, 'image_id': image_id, 'category_id': 1,
                             'bbox': [float(xmin), float(ymin), float(w), float(h)],
                             'area': float(w * h), 'iscrowd': 0})
            image_id += 1
            ann_id += 1

        if i % 10 == 0:
            print(f'Generated {i+1}/{args.num} images')

    write_coco(outdir, images_meta, ann_meta)
    print('Done. Images ->', images_dir)
    print('Labels ->', labels_dir)
    print('COCO ->', outdir / 'coco_annotations.json')


if __name__ == '__main__':
    main()
