## Solving a classification problem(?)

Descripción

Libraries
- template: cookiecutter-datascience
- ml stack: sklearn, category_encoders, pandas

Flujo ML: 
1. Analyze the dataset
    1. ¿Feature interaction y posible selección?
2. Lean Thinking: Build-Measure-Learn our models.  
    Andrew Ng advice: [here](https://www.coursera.org/lecture/machine-learning-projects/build-your-first-system-quickly-then-iterate-jyWpn)
3. Repetitive steps: Pipelines
    Take care (and prevent) data leakage [here](https://towardsdatascience.com/pre-process-data-with-pipeline-to-prevent-data-leakage-during-cross-validation-e3442cca7fdc)
4. Select a model at a glance
5. Tuning Selected model: GridSearch
6. Evaluate
    1. Metric: accuracy
    2. Classification Report
    3. Confusion Matrix
7. Train with whole Dataset & Submit!

Future (mejoras y otras cosas) enhancements and other stuffs
- Handling imbalanced datasets
- Scaling numeric data
- Parallelize Approach?: Dask! (anti-memorybased)
- Ensemble models: mlextend
- Feature Selection
