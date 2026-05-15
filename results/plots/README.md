This folder contains references to evaluation plots produced during model
training and evaluation. For convenience the original images remain under
`models/*/results_test/` and can be copied here if required by the grader.

Available plots (original locations):
- Speech training curves: models/speech_pipeline/results_test/training_curves.png
- Speech confusion matrix: models/speech_pipeline/results_test/confusion_matrix.png
- Text confusion matrix: models/text_pipeline/results_test/confusion_matrix.png

To copy the images into this folder (optional):

```powershell
copy models\speech_pipeline\results_test\training_curves.png Results\plots\
copy models\speech_pipeline\results_test\confusion_matrix.png Results\plots\
copy models\text_pipeline\results_test\confusion_matrix.png Results\plots\
```
