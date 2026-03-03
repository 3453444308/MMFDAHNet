import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from model import MultiBandFeatureExtractor, DomainDiscriminator, SpatialCognitivePredictor
from datasets import data_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CrossSubjectEEGDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, is_source=True):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        domain_val = 0.0 if is_source else 1.0
        domain_arr = np.full(len(labels), domain_val, dtype=np.float32)
        self.domain_labels = torch.tensor(domain_arr, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].reshape(16, 256, 3)
        y = self.labels[idx]
        domain_label = self.domain_labels[idx]
        return x, y, domain_label


def evaluate_model_accuracy(feature_extractor, classifier, test_loader):
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_data, test_labels, _ in test_loader:
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            features = feature_extractor(test_data)
            logits = classifier(features)
            pred = logits.argmax(dim=1)
            correct += pred.eq(test_labels).sum().item()
            total += len(test_labels)
    return correct / total if total > 0 else 0.0


def evaluate_comprehensive_metrics(feature_extractor, classifier, test_loader):
    feature_extractor.eval()
    classifier.eval()

    y_true, y_pred, y_prob = [], [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for test_data, test_labels, _ in test_loader:
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            features = feature_extractor(test_data)
            logits = classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            loss = criterion(logits, test_labels)
            running_loss += loss.item()

            y_true.extend(test_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(y_true, [prob[1] for prob in y_prob]),
        'y_true': y_true,
        'y_pred': y_pred,
        'loss': running_loss
    }
    return metrics


def train_domain_adaptation(source_loader, target_loader, epochs=200, phase1_epochs=100, lambda_weight=0.2):
    feature_extractor = MultiBandFeatureExtractor().to(device)
    classifier = SpatialCognitivePredictor().to(device)
    domain_discriminator = DomainDiscriminator(input_size=288, output_size=2).to(device)

    opt_phase1 = torch.optim.Adam([
        {"params": feature_extractor.parameters()},
        {"params": classifier.parameters()}
    ], lr=0.0001)

    opt_phase2 = torch.optim.Adam([
        {"params": feature_extractor.parameters()},
        {"params": classifier.parameters()},
        {"params": domain_discriminator.parameters()}
    ], lr=0.0001)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()

        total_loss, total_class_loss, total_domain_loss = 0.0, 0.0, 0.0
        source_correct, total_source = 0, 0
        domain_correct, total_domain = 0, 0

        target_iter = itertools.cycle(target_loader)

        for source_data, source_labels, source_domain_labels in source_loader:
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            source_domain_labels = source_domain_labels.to(device)

            target_data, _, target_domain_labels = next(target_iter)
            target_data = target_data.to(device)
            target_domain_labels = target_domain_labels.to(device)

            if epoch < phase1_epochs:
                opt_phase1.zero_grad()

                source_features = feature_extractor(source_data)
                source_logits = classifier(source_features)

                loss_c = class_criterion(source_logits, source_labels)

                pred = source_logits.argmax(dim=1)
                source_correct += pred.eq(source_labels).sum().item()
                total_source += source_labels.size(0)

                loss_c.backward()
                opt_phase1.step()
                total_loss += loss_c.item()

            else:
                opt_phase2.zero_grad()

                all_data = torch.cat([source_data, target_data], dim=0)
                all_features = feature_extractor(all_data)

                source_features = all_features[:source_data.size(0)]

                source_logits = classifier(source_features)
                loss_c = class_criterion(source_logits, source_labels)

                class_pred = source_logits.argmax(dim=1)
                source_correct += class_pred.eq(source_labels).sum().item()
                total_source += source_labels.size(0)

                alpha = 1.0
                domain_logits = domain_discriminator(all_features, alpha)
                domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)

                loss_d = domain_criterion(domain_logits, domain_labels)

                domain_pred = domain_logits.argmax(dim=1)
                domain_correct += domain_pred.eq(domain_labels).sum().item()
                total_domain += domain_labels.size(0)

                loss_total = loss_c + lambda_weight * loss_d

                total_class_loss += loss_c.item()
                total_domain_loss += loss_d.item()

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=1.0)
                opt_phase2.step()

                total_loss += loss_total.item()

        target_acc = evaluate_model_accuracy(feature_extractor, classifier, target_loader)

        if epoch < phase1_epochs:
            source_acc = source_correct / total_source if total_source > 0 else 0.0
            print(
                f"Epoch {epoch + 1}/{epochs} [Phase 1] | Class Loss: {total_loss:.4f} | Source Acc: {source_acc:.4f} | Test Acc: {target_acc:.4f}")
        else:
            domain_acc = domain_correct / total_domain if total_domain > 0 else 0.0
            class_acc = source_correct / total_source if total_source > 0 else 0.0
            print(
                f"Epoch {epoch + 1}/{epochs} [Phase 2] | Class Loss: {total_class_loss:.4f} | Domain Loss: {total_domain_loss:.4f} | Class Acc: {class_acc:.4f} | Domain Acc: {domain_acc:.4f} | Test Acc: {target_acc:.4f}")

    test_metrics = evaluate_comprehensive_metrics(feature_extractor, classifier, target_loader)
    print(f"Final Test Accuracy for this fold: {test_metrics['accuracy']:.4f}")

    return feature_extractor, classifier, test_metrics


if __name__ == "__main__":
    batchsize = 64
    n_folds = 16

    results = []
    all_y_true = []
    all_y_pred = []

    data, labels, groups = data_process()
    logo = LeaveOneGroupOut()

    for fold_idx, (source_idx, target_idx) in enumerate(logo.split(data, labels, groups=groups)):
        print(f'\n{"=" * 15} Fold {fold_idx + 1}/{n_folds} {"=" * 15}')

        x_source, y_source = data[source_idx, :, :], labels[source_idx]
        x_target, y_target = data[target_idx, :, :], labels[target_idx]

        set_seed(42 + fold_idx)

        source_dataset = CrossSubjectEEGDataset(x_source, y_source, is_source=True)
        source_loader = DataLoader(source_dataset, batch_size=batchsize, shuffle=True, worker_init_fn=worker_init_fn)

        target_dataset = CrossSubjectEEGDataset(x_target, y_target, is_source=False)
        target_loader = DataLoader(target_dataset, batch_size=batchsize, shuffle=True, worker_init_fn=worker_init_fn)

        feature_extractor, classifier, test_metrics = train_domain_adaptation(
            source_loader=source_loader,
            target_loader=target_loader,
            epochs=200,
            phase1_epochs=100,
            lambda_weight=0.2
        )

        results.append({
            'Fold': fold_idx + 1,
            'Accuracy': test_metrics['accuracy'],
            'F1 Score': test_metrics['f1'],
            'Recall': test_metrics['recall'],
            'Precision': test_metrics['precision'],
            'AUC': test_metrics['auc']
        })

        all_y_true.extend(test_metrics['y_true'])
        all_y_pred.extend(test_metrics['y_pred'])

    results_df = pd.DataFrame(results)
    results_df.to_excel('results.xlsx', index=False)

    print('\n' + '=' * 15 + ' Final Cross-Subject Results ' + '=' * 15)
    print(f'Average Accuracy:  {results_df["Accuracy"].mean():.4f} ± {results_df["Accuracy"].std():.4f}')
    print(f'Average F1 Score:  {results_df["F1 Score"].mean():.4f} ± {results_df["F1 Score"].std():.4f}')
    print(f'Average Recall:    {results_df["Recall"].mean():.4f} ± {results_df["Recall"].std():.4f}')
    print(f'Average Precision: {results_df["Precision"].mean():.4f} ± {results_df["Precision"].std():.4f}')
    print(f'Average AUC:       {results_df["AUC"].mean():.4f} ± {results_df["AUC"].std():.4f}')

    print("\nOverall Confusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)