# import joblib

# from pathlib import Path


# from sklearn.tree import (
#     DecisionTreeClassifier,
#     plot_tree
# )


# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )


# import matplotlib.pyplot as plt
# import seaborn as sns


# from preprocessing.data_preprocessing import prepare_tree_data



# # Project directories

# BASE_DIR = Path(__file__).resolve().parent.parent

# PLOT_DIR = BASE_DIR / "plots"

# MODEL_DIR = BASE_DIR / "saved_models"


# PLOT_DIR.mkdir(exist_ok=True)

# MODEL_DIR.mkdir(exist_ok=True)





# # Load unscaled tree data

# def load_data():


#     data = prepare_tree_data()


#     return data





# # Find best tree depth

# def find_best_depth(
#         X_train,
#         y_train,
#         X_val,
#         y_val
# ):


#     depths = [

#         2,
#         3,
#         5,
#         7,
#         10,
#         15,
#         None

#     ]


#     results = {}



#     for depth in depths:


#         model = DecisionTreeClassifier(

#             max_depth=depth,

#             random_state=42

#         )


#         model.fit(

#             X_train,

#             y_train

#         )



#         prediction = model.predict(

#             X_val

#         )



#         accuracy = accuracy_score(

#             y_val,

#             prediction

#         )


#         results[depth] = accuracy



#         print(

#             f"Depth={depth} Validation Accuracy={accuracy:.4f}"

#         )




#     best_depth = max(

#         results,

#         key=results.get

#     )


#     print(

#         "\nBest Depth:",

#         best_depth

#     )


#     return best_depth





# # Train final tree

# def train_tree(

#         X_train,

#         y_train,

#         best_depth

# ):


#     model = DecisionTreeClassifier(

#         max_depth=best_depth,

#         random_state=42

#     )


#     model.fit(

#         X_train,

#         y_train

#     )


#     return model





# # Evaluate model

# def evaluate_model(

#         model,

#         X_train,

#         y_train,

#         X_test,

#         y_test

# ):


#     train_prediction = model.predict(

#         X_train

#     )


#     test_prediction = model.predict(

#         X_test

#     )



#     train_accuracy = accuracy_score(

#         y_train,

#         train_prediction

#     )


#     test_accuracy = accuracy_score(

#         y_test,

#         test_prediction

#     )



#     print("\nTraining Accuracy:")

#     print(

#         train_accuracy * 100,

#         "%"

#     )



#     print("\nTesting Accuracy:")

#     print(

#         test_accuracy * 100,

#         "%"

#     )



#     print("\nClassification Report")


#     print(

#         classification_report(

#             y_test,

#             test_prediction

#         )

#     )



#     cm = confusion_matrix(

#         y_test,

#         test_prediction

#     )



#     plt.figure(

#         figsize=(6,5)

#     )



#     sns.heatmap(

#         cm,

#         annot=True,

#         fmt="d",

#         cmap="Blues",

#         xticklabels=[

#             "Obese",

#             "Fit"

#         ],

#         yticklabels=[

#             "Obese",

#             "Fit"

#         ]

#     )


#     plt.xlabel(

#         "Predicted"

#     )


#     plt.ylabel(

#         "Actual"

#     )


#     plt.title(

#         "Decision Tree Confusion Matrix"

#     )


#     plt.savefig(

#         PLOT_DIR / "decision_tree_confusion_matrix.png"

#     )


#     plt.close()



#     return test_accuracy





# # Visualize tree

# def visualize_tree(

#         model,

#         feature_names

# ):


#     plt.figure(

#         figsize=(14,8)

#     )


#     plot_tree(

#         model,

#         feature_names=feature_names,

#         class_names=[

#             "Obese",

#             "Fit"

#         ],

#         filled=True

#     )


#     plt.savefig(

#         PLOT_DIR / "decision_tree_structure.png",

#         dpi=300,

#         bbox_inches="tight"

#     )


#     plt.close()





# # Save model

# def save_model(

#         model,

#         best_depth,

#         accuracy

# ):


#     metadata = {


#         "algorithm":

#         "Decision Tree",


#         "best_depth":

#         best_depth,


#         "features":

#         [

#             "height_cm",

#             "weight_kg"

#         ],


#         "accuracy":

#         accuracy


#     }



#     joblib.dump(

#         model,

#         MODEL_DIR / "decision_tree.pkl"

#     )



#     joblib.dump(

#         metadata,

#         MODEL_DIR / "decision_tree_metadata.pkl"

#     )


#     print(

#         "\nDecision Tree saved!"

#     )







# # Main

# if __name__ == "__main__":



#     (

#         X_train,

#         X_val,

#         X_test,

#         y_train,

#         y_val,

#         y_test,

#         feature_names

#     ) = load_data()





#     best_depth = find_best_depth(

#         X_train,

#         y_train,

#         X_val,

#         y_val

#     )





#     model = train_tree(

#         X_train,

#         y_train,

#         best_depth

#     )





#     accuracy = evaluate_model(

#         model,

#         X_train,

#         y_train,

#         X_test,

#         y_test

#     )





#     visualize_tree(

#         model,

#         feature_names

#     )





#     save_model(

#         model,

#         best_depth,

#         accuracy

#     )