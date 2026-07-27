## Setup
Additional dataset — ASHRAE GEPIII (Kaggle, needs auth)

**Accept the competition rules** — required or downloads return HTTP 403:
   visit https://www.kaggle.com/competitions/ashrae-energy-prediction/rules,
   click "Late Submission", then "I Understand and Accept".

Download the three files we use (~686 MB; we skip test.csv / sample_submission.csv):

```bash
mkdir -p submission/dataset_ashrae/raw
for f in building_metadata.csv weather_train.csv train.csv; do
  .venv/bin/kaggle competitions download -c ashrae-energy-prediction -f $f -p submission/dataset_ashrae/raw
done
cd submission/dataset_ashrae/raw && for z in *.zip; do unzip -o "$z" && rm "$z"; done
```

Then convert the raw ASHRAE files into our pipeline's schema:  <!-- TODO: converter script, coming next -->

```bash
# ../.venv/bin/python submission/convert_ashrae.py   (to be added)
```
