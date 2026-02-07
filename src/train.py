import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys

sys.path.append(".")

from features import extract_features


def load_data():
    X = np.loadtxt(r"C:\Users\satya\on-device-activity-recognition\data\raw\UCI HAR Dataset\train\Inertial Signals\body_acc_x_train.txt")
    y = np.loadtxt(r"C:\Users\satya\on-device-activity-recognition\data\raw\UCI HAR Dataset\train\y_train.txt").astype(int)
    return X, y


def build_feature_matrix(X):
    return np.array([extract_features(signal) for signal in X])


def main():
    X_raw, y = load_data()
    X_features = build_feature_matrix(X_raw)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_features, y)

    y_pred = model.predict(X_features)

    print(classification_report(y, y_pred))


if __name__ == "__main__":
    main()
