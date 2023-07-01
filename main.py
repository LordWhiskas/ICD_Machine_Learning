from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from ICD import ICD
from keras.datasets import mnist

def main():
    """
    Main function to demonstrate the usage of the ICD algorithm for classification tasks.
    Loads the Iris and MNIST datasets, trains the ICD algorithm on them, and evaluates the performance.
    Finally, visualizes the prototypes using PCA for dimensionality reduction.
    """

    # Load the Iris dataset and train the ICD algorithm on it
    iris_dataset = load_iris()
    dr = PCA(n_components=2)
    icd_iris = ICD(iris_dataset, "iris", 115)
    icd_iris.initialize_prototypes()
    icd_iris.update_prototypes()

    # Evaluate the performance of the ICD algorithm on the Iris dataset
    accuracy, precision, returnability, f1, recall = icd_iris.evaluate_model()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Returnability: {returnability:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    # Visualize the prototypes of the ICD algorithm on the Iris dataset
    icd_iris.visualize_prototypes(dr)

    # Load the MNIST dataset and train the ICD algorithm on it
    icd_mnist = ICD(mnist, "mnist", 1000)
    icd_mnist.initialize_prototypes()
    icd_mnist.update_prototypes()
    
    # Evaluate the performance of the ICD algorithm on the MNIST dataset
    accuracy, precision, returnability, f1, recall = icd_mnist.evaluate_model()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Returnability: {returnability:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    # Visualize the prototypes of the ICD algorithm on the MNIST dataset
    icd_mnist.visualize_prototypes(dr)

if __name__ == "__main__":
    main()
