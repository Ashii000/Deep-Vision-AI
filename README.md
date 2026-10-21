# DeepVision AI
### Industrial Visual Inspection with Deep Learning
**Version:** 1.0 • **Modes:** A (classification) + B (anomaly heatmap)

This project combines two inspection workflows in one Streamlit app:
- **Mode A — Defect classification:** a ResNet18 transfer-learning classifier predicts labels learned from your own labeled dataset.
- **Mode B — Anomaly detection:** a pretrained ResNet18 feature extractor compares the uploaded image with normal reference images and visualizes spatial differences as a heatmap.
- **Combined inspection:** runs both modes when the required files/data are present.

> This is an educational prototype. It is not a certified factory inspection system. Scores depend on data and should not be presented as guaranteed defect probabilities.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet for first-time package/model downloads
- CPU can run small tests; GPU is helpful but not mandatory
- A browser

## Fast setup on a lab PC

1. Extract the ZIP to a folder, e.g. `DeepVisionAI`.
2. Open Command Prompt/Terminal in that folder.
3. Create and activate a virtual environment (recommended):

Windows:
```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
5. Start the app:
```bash
streamlit run app.py
```
6. The browser opens. Upload a JPG/PNG image.

### If PyTorch installation fails
Use the official PyTorch installation selector for your PC/OS and CPU/GPU setup, then install the matching `torch` and `torchvision` versions. Keep them compatible. Afterward install the remaining packages:
```bash
pip install streamlit numpy Pillow matplotlib
```

## Mode A: Train your own classifier

The model needs labeled examples; the app does not magically know what a defective product is without training data.

Put images in this structure:
```text
data/classifier/
  train/
    normal/
      good_01.jpg
      good_02.jpg
    defective/
      defect_01.jpg
      defect_02.jpg
  val/
    normal/
    defective/
```
- `normal` and `defective` are example class names; you may use other class names, but keep train and val folders identical.
- Use images of the same product type and similar inspection conditions.
- Include multiple examples, backgrounds, angles, lighting conditions, and defect types.
- Keep validation images separate from training images.
- For a meaningful demo, aim for dozens to hundreds of images per class if feasible. The tiny minimum in the script only checks that the pipeline can run; it does not guarantee a good model.

Run:
```bash
python scripts/train_classifier.py
```
The script fine-tunes the final layer of a pretrained ResNet18 and saves `models/classifier.pt`.
Then refresh the Streamlit app and choose **A — Defect classification** or **Combined inspection**.

### Where to get data
For a classroom prototype, use a reputable, labeled dataset such as MVTec AD or a suitable public industrial-defect image dataset. Check its license and class structure before using it. If adapting MVTec AD, note that it is primarily designed for anomaly detection and may need extra organization/labels to train the supervised classifier.

## Mode B: Anomaly detection + heatmap

Put at least two **normal/good** reference images into:
```text
data/anomaly/reference_normal/
  normal_01.jpg
  normal_02.jpg
  normal_03.jpg
```
For best results use a larger, varied set of normal images of the same product/category and similar camera/lighting conditions. The uploaded image is compared with the normal reference feature template. Regions that differ more are shown in the heatmap.

This is a lightweight feature-distance baseline using a pretrained ResNet18. It is **not** the full PatchCore algorithm, and the score is relative rather than a calibrated probability. A heatmap can highlight harmless differences (lighting, background, angle) as well as possible defects.

## Troubleshooting
- **Classifier model not found:** add labeled data and run `python scripts/train_classifier.py`.
- **Anomaly mode says reference images missing:** add at least two good images to `data/anomaly/reference_normal/`.
- **First run is slow:** pretrained weights may be downloading; check internet.
- **Out of memory:** use smaller images, close other apps, or run CPU mode.
- **No GPU:** CPU mode is supported but slower.
- **Windows terminal says `streamlit` is not recognized:** run `python -m streamlit run app.py`.
- **Module not found:** activate your environment and run `pip install -r requirements.txt`.

## Suggested 4-day plan (2–3 hours/day)
**Day 1:** Setup; run app; download model weights; test Mode B with normal reference images.
**Day 2:** Collect/organize labeled data and train Mode A; review output and limitations.
**Day 3:** Test both modes on new images; capture screenshots; fix any setup issues.
**Day 4:** Prepare architecture diagram, evaluation results, README, and presentation.

## Architecture
```text
Uploaded image
     |
     +--> Mode A: resize + normalize --> ResNet18 classifier --> class + probabilities
     |
     +--> Mode B: resize + normalize --> pretrained ResNet18 feature map
                                      --> compare with normal reference template
                                      --> anomaly score + heatmap overlay
     |
     +--> Streamlit result dashboard
```

## What to explain in viva
1. Why transfer learning saves time compared with training a CNN from scratch.
2. Difference between classification (A) and anomaly detection/localization (B).
3. Why labeled examples are needed for supervised classification.
4. Why anomaly detection can use only normal reference images.
5. Why confidence/anomaly scores are not guarantees.
6. How you would improve reliability: larger dataset, separate test set, precision/recall/F1 for A, and pixel-level/region metrics for B.
7. Why lighting, product angle, background and dataset imbalance can affect results.

## Important implementation note
The project ZIP includes source code and empty dataset folders, not copyrighted datasets or pretrained weights. The pretrained weights download on first use when internet is available. The classifier model file is created after training. Keep your data and generated `models/classifier.pt` backed up if working on a temporary lab PC.
