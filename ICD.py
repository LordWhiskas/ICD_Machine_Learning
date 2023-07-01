# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.animation as animation

# Define a class for the ICD algorithm
class ICD:
    
    # Constructor to initialize the dataset, name, and subset size (if any)
    def __init__(self, dataset, name, subset_size=None):
        self.name = name
        print(self.name)
        
        # If the dataset is iris, use StandardScaler to normalize the data
        if self.name == "iris":
            self.X = dataset.data
            self.Y = dataset.target
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
        
        # If the dataset is mnist, concatenate the train and test data and normalize it
        else:
            (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
            self.X = np.concatenate((train_images, test_images), axis=0)
            self.Y = np.concatenate((train_labels, test_labels), axis=0)
            self.X = self.X / 255.0
        
        # If subset size is specified, take only the first subset_size examples
        if subset_size is not None:
            self.X = self.X[:subset_size]
            self.Y = self.Y[:subset_size]
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        
        # Create an empty dictionary to hold the prototypes
        self.prototypes = {}
        
        # Set the distance limit for creating new prototypes
        self.distance_limit = 0.9
    
    # Method to initialize the prototypes with the first example of each class  
    def initialize_prototypes(self):
        for data, label in zip(self.X_train, self.y_train):
            if label not in self.prototypes:
                self.prototypes[label] = {'examples': [data], 'count': 1}
    
    # Method to update the prototypes based on the training data
    def update_prototypes(self):
        total_data = len(self.X_train)
        for idx, (data, label) in enumerate(zip(self.X_train, self.y_train)):
            # Find the closest prototype in the same class
            prototype = None
            min_distance = float('inf')
            for i, example in enumerate(self.prototypes[label]['examples']):
                distance = self.euclidean_distance(data, example)
                if distance < min_distance:
                    min_distance = distance
                    prototype = (i, example)
            
            # Check if distance is within the limit, otherwise add a new prototype
            if min_distance <= self.distance_limit:
                updated_prototype = (prototype[1] * self.prototypes[label]['count'] + data) / (self.prototypes[label]['count'] + 1)
                self.prototypes[label]['examples'][prototype[0]] = updated_prototype
                self.prototypes[label]['count'] += 1
            else:
                self.prototypes[label]['examples'].append(data)
                self.prototypes[label]['count'] += 1
            
            # Print the progress
            progress = (idx + 1) / total_data * 100
            print(f"Processed {progress:.2f}% of the dataset")

    # Method to compute the Euclidean distance between two points
    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y)**2))
    
    # Method to predict the class of a given example
    def predict_class(self, example):
        min_distance = float('inf')
        predicted_class = None
        for C, prototype in self.prototypes.items():
            for prototype_example in prototype['examples']:
                distance = self.euclidean_distance(example, prototype_example)
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = C
        return predicted_class
    
    # Method to evaluate the performance of the model on the test set
    def evaluate_model(self):
        y_pred = []
        for example in self.X_test:
            predicted_class = self.predict_class(example)
            y_pred.append(predicted_class)

        # Calculate the accuracy, precision, recall, F1 score, and confusion matrix    
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

        cm = confusion_matrix(self.y_test, y_pred)
        
        # Calculate the true positives (tp) and false positives (fp) for each class
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        
        # Calculate the returnability metric
        returnability = np.mean(tp / (tp + fp))

        return accuracy, precision, returnability, f1, recall
    
    # Method to visualize the prototypes using dimensionality reduction
    def visualize_prototypes(self, dr):
        # Reshape the data if it has more than 2 dimensions
        X = self.X.reshape((self.X.shape[0], -1)) if len(self.X.shape) > 2 else self.X
        X_dr = dr.fit_transform(X)
        
        # Set the color map and create the figure and axis for the plot
        colormap = plt.cm.get_cmap("viridis", len(np.unique(self.Y)))
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Adjust the left, bottom, width, and height of the plot

        # Define the animation function to plot the data and prototypes
        def animate(i):
            ax.clear()

            # Plot the data points with their class colors
            for idx, C in enumerate(np.unique(self.Y)):
                ax.scatter(X_dr[self.Y == C, 0], X_dr[self.Y == C, 1], c=[colormap(idx)], label=f"Class {C}")

            # Plot the prototypes for each class
            for j in range(i + 1):
                current_prototypes = prototypes_dr[class_start_indices[j]:class_start_indices[j + 1]]
                current_labels = prototypes_labels[class_start_indices[j]:class_start_indices[j + 1]]

                for idx, (example, label) in enumerate(zip(current_prototypes, current_labels)):
                    ax.scatter(example[0], example[1], c=[colormap(label)], marker='*', s=200, edgecolors='k', linewidths=1, label=f"Prototype Class {label}" if idx == 0 else "")

            # Set the axis labels, legend, and title        
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend to the right side
            ax.set_title(f'ICD Algorithm Separation of {self.name.capitalize()} Dataset Classes')

            return ax,

        # Extract the prototypes and their labels from the dictionary
        prototypes_array = []
        prototypes_labels = []
        class_start_indices = [0]
        for label, prototype in self.prototypes.items():
            examples = np.array(prototype['examples'])
            examples_reshaped = examples.reshape((examples.shape[0], -1)) if len(examples.shape) > 2 else examples
            prototypes_array.extend(examples_reshaped)
            prototypes_labels.extend([label] * len(examples_reshaped))
            class_start_indices.append(len(prototypes_labels))

        # Convert the prototypes to a numpy array and reduce their dimensionality using the given dimensionality reduction algorithm
        prototypes_array = np.array(prototypes_array)
        prototypes_dr = dr.transform(prototypes_array)

        # Create the animation and display it
        ani = animation.FuncAnimation(fig, animate, frames=len(class_start_indices) - 1, interval=2000, blit=False)
        plt.show()