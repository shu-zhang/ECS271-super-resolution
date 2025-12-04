echo "Running ResNet"
python3 src/training_resnet.py

echo "Running swinir"
python3 src/training_swinir.py

echo "Evaluate models"
python3 src/evaluate_models.py