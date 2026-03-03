import torch
import numpy as np
from torch.utils.data import DataLoader

# Import necessary modules from your existing files
from train import CrossSubjectEEGDataset, evaluate_comprehensive_metrics
from model import MultiBandFeatureExtractor, SpatialCognitivePredictor

# Configure the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_sub6_with_saved_weights():
    print(">>> Start loading weights and testing Subject 6 data...")

    # 1. 直接加载处理好的 .npy 测试数据 (避免数据泄露和重复标准化)
    data_path = 'sub6_data.npy'
    label_path = 'sub6_labels.npy'

    try:
        x_target = np.load(data_path)
        y_target = np.load(label_path)
        print(f">>> Successfully loaded data. Samples: {len(x_target)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Please ensure {data_path} and {label_path} exist in the directory.")

    # Build DataLoader for testing
    # is_source is set to False for target domain
    target_dataset = CrossSubjectEEGDataset(x_target, y_target, is_source=False)
    # No shuffle needed during the testing phase
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=False)

    # 2. Initialize network models and move to device
    feature_extractor = MultiBandFeatureExtractor().to(device)
    classifier = SpatialCognitivePredictor().to(device)

    # 3. Load pre-trained weights
    # 确保这里的路径指向你正确训练出来的 sub6 留一验证权重
    extractor_path = 'feature_extractor_sub6.pth'
    classifier_path = 'classifier_sub6.pth'

    try:
        feature_extractor.load_state_dict(torch.load(extractor_path, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print(">>> Successfully loaded pre-trained weights!")
    except FileNotFoundError:
        raise FileNotFoundError("Model weight files not found. Please check the 'weights/' directory.")

    # Set the models to evaluation mode (Very important: turns off Dropout and BatchNorm tracking)
    feature_extractor.eval()
    classifier.eval()

    # 4. Call the comprehensive evaluation function in train.py for testing
    print(">>> Performing comprehensive metrics evaluation...")
    test_metrics = evaluate_comprehensive_metrics(feature_extractor, classifier, target_loader)

    # 5. Print test results
    print('\n' + '=' * 15 + ' Subject 6 Test Results ' + '=' * 15)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"AUC:       {test_metrics['auc']:.4f}")

    print("\nConfusion Matrix:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
    print(cm)


if __name__ == "__main__":
    test_sub6_with_saved_weights()