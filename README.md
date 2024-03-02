

User
https://github.com/dariusstrongman/Project-Build-an-ML-Pipeline-Starter.git
GPT
Talked to pluginlab.askthecode.dsomok.online
The repository "Project-Build-an-ML-Pipeline-Starter" on GitHub has a main branch containing various files and directories, which are structured as follows:

GitHub workflows, license, and README files at the root level.
A components directory containing multiple ML components, each with its own MLproject file, conda.yml for environment setup, and Python scripts. It also includes utility scripts for Weights & Biases (wandb) integration.
Configuration files such as MLproject, conda.yml, config.yaml, and environment.yml in the root directory for setting up the ML project and environment.
An images directory with images related to the project documentation.
The main.py script, presumably the main entry point for the project.
A src directory with subdirectories for various stages of the ML pipeline, like basic_cleaning, data_check, eda (Exploratory Data Analysis with a Jupyter notebook), and train_random_forest. Each of these subdirectories contains an MLproject file, conda.yml, and Python scripts relevant to the stage.
This structure suggests a modular ML project setup using MLflow, with separate components for data preparation, exploratory data analysis, model training, etc., and integration with wandb for experiment tracking.

Is there a specific part of the repository you would like to explore further or any specific task you need assistance with?


1 / 2

User
Project Environment Set Up
Create Environment Locally
Download and install Conda if you don't have it already.

Make sure you are in the Project-Build-an-ML-Pipeline-Starter directory. Make sure to have conda installed and ready, then create a new environment using the environment.yml file provided in the root of the repository and activate it (note, this process may take a while):

> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
Make sure you are using Python 3.9.15 installed in the nyc_airbnb_dev environment. You can check it by this command:

> which python
You should see a message similar to:

/opt/anaconda3/envs/nyc_airbnb_dev/bin/python
If you have trouble using Python in the nyc_airbnb_dev environment, try:

> export PATH=[your path of the enviornment]/bin:$PATH
For example:

> export PATH=/opt/anaconda3/envs/nyc_airbnb_dev/bin:$PATH
To get the path of the nyc_airbnb_dev environment, try the following command, and all the environments' paths should be listed.

> conda info -e
Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to (opens in a new tab)https://wandb.ai/authorize(opens in a new tab) and click on the + icon (copy to clipboard), then paste your key into this command:

> wandb login [your API key]
You should see a message similar to this:

wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
Jupyter Lab
Jupyter Lab is installed in the environment above. To start the Jupyter Lab, use the following command in the terminal.

> jupyter-lab
GPT
To set up the project environment locally for the "Project-Build-an-ML-Pipeline-Starter", you can follow these steps:

Install Conda: If you haven't already, download and install Conda. You can get it from Conda's official website.

Navigate to the Project Directory: Open a terminal and change the directory to where the "Project-Build-an-ML-Pipeline-Starter" repository is cloned.

Create and Activate the Conda Environment:

To create a new environment using the environment.yml file located in the root of the repository, run:
bash
Copy code
conda env create -f environment.yml
Once the environment is created, activate it by running:
Copy code
conda activate nyc_airbnb_dev
Verify Python Installation:

To ensure you are using Python 3.9.15 installed in the nyc_airbnb_dev environment, run:

bash
Copy code
which python
You should see a path similar to /opt/anaconda3/envs/nyc_airbnb_dev/bin/python.

If you encounter any issues using Python in the nyc_airbnb_dev environment, you may need to adjust your PATH. For example:

bash
Copy code
export PATH=/opt/anaconda3/envs/nyc_airbnb_dev/bin:$PATH
To list all your Conda environments and their paths, you can use:

Copy code
conda info -e
Set Up Weights & Biases:

To use Weights & Biases (W&B) for tracking experiments, first, obtain your API key by visiting https://wandb.ai/authorize. Click on the + icon to copy your API key to the clipboard.

Login to W&B by running the following command in your terminal, replacing [your API key] with the actual key you copied:

css
Copy code
wandb login [your API key]
You should see a confirmation message indicating that the key was added successfully to your .netrc file.

Using Jupyter Lab:

Jupyter Lab is included in the environment you set up. To start Jupyter Lab, simply run:
Copy code
jupyter-lab
By following these steps, you should have a fully functional local development environment set up for working on the "Project-Build-an-ML-Pipeline-Starter".

User
Step 2: Data Cleaning
Now we transfer what we have done in EDA to a new "basic cleaning" step that cleans the sample.csv artifact and creates a new clean_sample.csv with the cleaned data.

Go to src/basic_cleaning, containing the files required for an MLflow step:

conda.yml: conda environment for the step
MLproject: parameters and definitions of the step
run.py: script of the step
Understand arguments in MLproject
Check the arguments stored in the src/basic_cleaning/MLproject. These are the arguments passed to the basic_cleaning step.

input_artifact: the input artifact
output_artifact: the name for the output artifact
output_type: the type for the output artifact
output_description: a description of the output artifact
min_price: the minimum price to consider
max_price: the maximum price to consider
All parameters should be of type str except min_price and max_price which should be float.

Add arguments information in run.py
Using the argument information you found above, fill in the missing type and description of the arguments in src/basic_cleaning/run.py.

Note the comments like TODO, INSERT TYPE HERE and INSERT DESCRIPTION HERE. Do not change code with comments like DO NOT MODIFY.

Add the step to the pipeline
Note the following comment in main.py. That is where you implement the code:

##################
# Implement here #
##################
Add the code below to the basic_cleaning step in the pipeline (the main.py file). You don't need to modify the code.

if "basic_cleaning" in active_steps:
    _ = mlflow.run(
         os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
         "main",
         parameters={
             "input_artifact": "sample.csv:latest",
             "output_artifact": "clean_sample.csv",
             "output_type": "clean_sample",
             "output_description": "Data with outliers and null values removed",
             "min_price": config['etl']['min_price'],
             "max_price": config['etl']['max_price']
         },
     )
Please note how the path to the step is constructed:

os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning")
This is necessary because Hydra executes the script in a different directory than the root of the starter kit. You will have to do the same for every step that you will add to the pipeline.

Remember that when you refer to an artifact stored on W&B, you MUST specify a version or a tag. For example, here, the input_artifact should be sample.csv:latest and NOT just sample.csv.

If you forget to do this, you will see a message like Attempted to fetch artifact without alias (e.g. "<artifact_name>:v3" or "<artifact_name>:latest")

Run the step
Run the pipeline with the following command:

> mlflow run . -P steps=basic_cleaning
If you go to W&B, you will see the new artifact type clean_sample and within it the clean_sample.csv artifact.
GPT
Talked to pluginlab.askthecode.dsomok.online
Step 2: Data Cleaning
Understanding arguments in src/basic_cleaning/MLproject
The MLproject file under src/basic_cleaning defines the following arguments for the basic cleaning step:

input_artifact: The initial artifact to be cleaned (Type: string)
output_artifact: The output artifact for cleaned data (Type: string)
output_type: The type of the output dataset (Type: string)
output_description: Description of the output dataset (Type: string)
min_price: Minimum house price to be considered (Type: float)
max_price: Maximum house price to be considered (Type: float)
Adding arguments information in src/basic_cleaning/run.py
In src/basic_cleaning/run.py, you'll need to fill in the missing type and description for each argument based on the information from MLproject. Here's how you can do it:

python
Copy code
parser.add_argument(
    "--input_artifact", 
    type=str,
    help="Initial artifact to be cleaned",
    required=True
)

parser.add_argument(
    "--output_artifact", 
    type=str,
    help="Output artifact for cleaned data",
    required=True
)

parser.add_argument(
    "--output_type", 
    type=str,
    help="Type of the output dataset",
    required=True
)

parser.add_argument(
    "--output_description", 
    type=str,
    help="Description of the output dataset",
    required=True
)

parser.add_argument(
    "--min_price", 
    type=float,
    help="Minimum house price to be considered",
    required=True
)

parser.add_argument(
    "--max_price",
    type=float,
    help="Maximum house price to be considered",
    required=True
)
Adding the step to the pipeline in main.py
In main.py, locate the section where you need to implement the basic cleaning step and add the following code:

python
Copy code
if "basic_cleaning" in active_steps:
    _ = mlflow.run(
        os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
        "main",
        parameters={
            "input_artifact": "sample.csv:latest",
            "output_artifact": "clean_sample.csv",
            "output_type": "clean_sample",
            "output_description": "Data with outliers and null values removed",
            "min_price": config['etl']['min_price'],
            "max_price": config['etl']['max_price']
        },
    )
This code snippet integrates the basic cleaning step into the pipeline by specifying the step's location, entry point, and parameters as defined in the MLproject file.

Remember, when referring to an artifact stored on Weights & Biases, you must specify a version or a tag, such as sample.csv:latest.

Running the step
To run the pipeline with the basic cleaning step, use the following command:

arduino
Copy code
mlflow run . -P steps=basic_cleaning
After running this command, you should see a new artifact type clean_sample in Weights & Biases, containing the clean_sample.csv artifact with cleaned data.

User
Step 3: Data Testing
After the cleaning, it is a good practice to put some tests that verify that the data does not contain surprises. In this section, you will work in src/data_check to write a few tests and add it to the ML pipeline.

Create a reference dataset
One of our tests will compare the distribution of the current data sample with a reference, to ensure that there is no unexpected change. Therefore, we first need to define a "reference dataset". We will just tag the latest clean_sample.csv artifact on W&B as our reference dataset.

Go to https://wandb.ai/home(opens in a new tab) in your web browser.
Navigate to your project, then to the artifact tab.
Click on "clean_sample", then on the version with the latest tag. This is the last one we produced in the previous basic cleaning step.
Add a tag reference to it by clicking the "+" in the Aliases section on the right.
Add a tag reference by clicking on the "+" button next to "Aliases" for the artifact overview.
Adding a tag

Create tests
Now we are ready to add some tests. In the starter kit, you can find a src/data_check step that you need to complete two tests.

test_row_count
test_price_range
Note the following comment in the src/data_check/test_data.py. That is where you add the tests.

########################################################
# Implement here test_row_count and test_price_range   #
########################################################
test_row_count
Let's start by appending to test_data.py the following test:

def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000
which checks that the size of the dataset is reasonable (not too small, not too large).

test_price_range
Now, add another test test_price_range(data, min_price, max_price) that checks that the price range is between min_price and max_price.

Hint: you can use the data['price'].between(...) method.

Also, remember that we are using closures, so the name of the variables that your test takes in MUST BE exactly data, min_price and max_price.

Add the step to the pipeline
Now add the data_check step to the main.py file, so that it gets executed as part of our pipeline.

Hint:

Check the basic_cleaning step you implemented before as an example. The implementation is very similar.
Check the src/data_check/MLproject for required arguments
Use clean_sample.csv:latest as csv and clean_sample.csv:reference as ref. Right now, they point to the same file, but later on, they will not: we will fetch another sample of data and therefore the latest tag will point to that.

Also, use the value stored in config.yaml for the other parameters. For example, use config["data_check"]["kl_threshold"] for the kl_threshold parameter.

Run the step
Run the pipeline and ensure the tests are executed and passed. Remember that you can run just this step with:

> mlflow run . -P steps=data_check
You can safely ignore the following DeprecationWarning if you see it:

DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' 
is deprecated since Python 3.3, and in 3.10 it will stop working
GPT
Talked to pluginlab.askthecode.dsomok.online
Step 3: Data Testing
After cleaning your data, it's crucial to verify its integrity and consistency. This step involves creating tests in the src/data_check directory and incorporating them into your ML pipeline.

Creating a Reference Dataset
To compare the distribution of your current data sample with a reference and ensure no unexpected changes have occurred, you first need to establish a "reference dataset." Follow these steps:

Go to your Weights & Biases project page.
Navigate to the "Artifacts" tab and locate the "clean_sample" artifact, specifically the version tagged as the latest.
Add a "reference" alias to this artifact to designate it as your reference dataset.
Implementing Tests
In the src/data_check directory, particularly in test_data.py, you'll add two tests:

test_row_count to ensure the dataset size is within a reasonable range (not too small or too large). Implement it as follows:

python
Copy code
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000
test_price_range to verify that prices are within the specified min_price and max_price range. Use the data['price'].between(...) method for this test. Remember, due to the usage of closures, the variable names must exactly match data, min_price, and max_price.

Adding the Data Check Step to the Pipeline
Next, incorporate the data_check step into your pipeline within main.py. Refer to the basic_cleaning implementation for guidance. The data_check step's inclusion should look something like this:

python
Copy code
if "data_check" in active_steps:
    _ = mlflow.run(
        os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
        "main",
        parameters={
            "csv": "clean_sample.csv:latest",
            "ref": "clean_sample.csv:reference",
            "kl_threshold": config["data_check"]["kl_threshold"],
            "min_price": config["etl"]["min_price"],
            "max_price": config["etl"]["max_price"]
        },
    )
This code snippet ensures the data_check step is executed as part of your pipeline. It uses the latest clean_sample.csv as the input CSV (csv) and the reference tagged clean_sample.csv as the reference CSV (ref). The other parameters, such as kl_threshold, min_price, and max_price, are fetched from the config.yaml.

Running the Data Check Step
To execute this step and validate your tests, use the command:

arduino
Copy code
mlflow run . -P steps=data_check
This command runs the data_check step of your ML pipeline, applying the tests you've defined to ensure data quality. If you encounter a DeprecationWarning related to the ABCs from 'collections', it can be safely ignored for now, as it's a warning for future deprecations in Python.

By following these steps, you've successfully added data validation tests to your ML pipeline, enhancing the robustness and reliability of your data processing workflow.

User
Step 4: Initial Training
Now the data is cleaned and checked. It's time to train the model. In this section, you will work in the following files and folders:

components/train_val_test_split: splitting data into training and test dataset
src/train_random_forest: construct pipelines to train a model and make inferences on the model
You will complete the code for these two steps and add them to the ML pipeline.

Data splitting
The step to split the training and testing dataset has been provided to you in components/train_val_test_split.

Add the step to the pipeline
Add this step to the main.py under the data_split step. You can see the parameters accepted by this step in components/train_val_test_split/MLproject.

Since this step is a component in the components folder, the path to the step can be expressed as:

_ = mlflow.run(
    f"{config['main']['components_repository']}/train_val_test_split",
    'main',
    parameters = {
             ...
    }
)
As usual, for parameters like test_size, random_seed and stratify_by, look at the modeling section in the config.yaml file. For input, you can use clean_sample.csv:latest.

Hint: The implementation of data_split is very similar to the download step.

Run the step
Now run the step.

After you execute, you will see something like:

2021-03-15 01:36:44,818 Uploading trainval_data.csv dataset
2021-03-15 01:36:47,958 Uploading test_data.csv dataset
This tells you that the script is uploading two new datasets: trainval_data.csv and test_data.csv.

If you go to W&B, you will see a new artifact type TEST_DATA and within it the test_data.csv artifact. And an artifact type TRAINVAL_DATA and within it the trainval_data.csv artifact.

Train Random Forest
Complete run.py
Read the script src/train_random_forest/run.py carefully and complete the following missing pieces.

Build a preprocessing pipeline that imputes missing values and encodes the variable
Build the inference pipeline called "sk_pipe"
Fit the "sk_pipe" pipeline
Save the "sk_pipe" pipeline
Save model metrics MAE and R2
All the places where you need to insert code are marked by a # YOUR CODE HERE comment and are delimited by two signs like ######################################. For example:

######################################
# Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
# YOUR CODE HERE
######################################
You can find further instructions in the run.py file.

Add the step to the pipeline
Once you are done, add this step to main.py under the train_random_forest step. As usual, you can check src/train_random_forest/MLproject for all the required arguments.

Hints:

The implementation should be similar to basic_cleaning and data_check steps.
Use trainval_data.csv:latest as trainval_artifact.
Use the name random_forest_export as output_artifact.
The main.py already provides a variable rf_config to be passed as the rf_config parameter.
Check the modeling section in config.yaml for the other parameters.
Run the step with hyperparameters optimization
Use the code below to run the training step with varying the hyperparameters of the Random Forest model. Note: this step may take a while to complete.

> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
This is done by exploiting the Hydra configuration system. It uses the multi-run feature (adding the -m option at the end of the hydra_options specification), and sets the parameter modeling.random_forest.max_depth to 10, 50, and the modeling.random_forest.n_estimators to 100, 200.
GPT
Talked to pluginlab.askthecode.dsomok.online
Step 4: Initial Training
Now that the data is cleaned and validated, the next step is to train the model. This involves working with data splitting and training components, and integrating them into the ML pipeline.

Data Splitting
The data splitting step is already provided in the components/train_val_test_split. You'll need to add this step to the main.py under the data_split step. The parameters accepted by this step are detailed in components/train_val_test_split/MLproject, which includes:

input: The input artifact to split (a CSV file)
test_size: The size of the test split, either as a fraction of the dataset or as a number of items
random_seed: The seed for the random number generator to ensure reproducibility
stratify_by: The column to use for stratification, if any
To add this step to the pipeline, use the following template in main.py, filling in the parameters as needed:

python
Copy code
if "data_split" in active_steps:
    _ = mlflow.run(
        f"{config['main']['components_repository']}/train_val_test_split",
        'main',
        parameters={
            "input": "clean_sample.csv:latest",
            "test_size": config['modeling']['test_size'],
            "random_seed": config['modeling']['random_seed'],
            "stratify_by": config['modeling']['stratify_by']
        }
    )
Train Random Forest
For the Random Forest training, you will need to complete the src/train_random_forest/run.py script. The script requires building a preprocessing pipeline, constructing the inference pipeline sk_pipe, fitting the pipeline, saving the trained model, and recording model metrics.

The missing pieces in run.py are marked with # YOUR CODE HERE comments. You'll need to:

Fit the sk_pipe pipeline with X_train and y_train.
Save the sk_pipe pipeline as an MLflow sklearn model in the directory random_forest_dir.
Save the R-squared score and MAE as model metrics.
Once you've completed the run.py, add the training step to main.py under the train_random_forest step, similar to how you added previous steps. Use trainval_data.csv:latest as the trainval_artifact and random_forest_export as the output_artifact. The other parameters can be fetched from the config.yaml.

Running the Step with Hyperparameter Optimization
To run the training step with hyperparameter optimization, use the following command:

bash
Copy code
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
This command varies the max_depth and n_estimators parameters of the Random Forest model to find the best combination. The -m option enables Hydra's multi-run feature, allowing for multiple configurations to be tested in parallel.

By following these steps, you'll complete the initial training phase of your ML pipeline, including data splitting and Random Forest training with hyperparameter optimization.

User
how do I get this file:  trainval_data.csv:latest
GPT
To obtain the trainval_data.csv:latest file, you would typically follow these steps within the context of a machine learning pipeline that uses Weights & Biases (W&B) for artifact tracking:

Navigate to your Weights & Biases Project: Log in to your W&B account and go to the project where you're tracking your ML pipeline artifacts.

Find the Artifact: In your W&B project dashboard, locate the "Artifacts" section, which lists all the artifacts produced by your pipeline runs. Look for an artifact named trainval_data.csv or similar, as the naming convention might slightly vary based on your pipeline's configuration.

Select the Latest Version: Artifacts in W&B can have multiple versions. Each time an artifact is logged with the same name, a new version is created. Look for the version tagged as latest. This tag is usually automatically assigned to the most recent version of an artifact.

Download the Artifact: Once you've found the trainval_data.csv:latest artifact, click on it to view its details. There should be an option to download the artifact directly from the W&B interface. The download option is typically represented by a download icon or a direct link.

If you are working within a script or a pipeline and want to programmatically access this artifact, you can use the W&B API. Here's a Python snippet that demonstrates how to do this:

python
Copy code
import wandb

# Initialize a W&B run
run = wandb.init(project='your_project_name', entity='your_wandb_username')

# Use the artifact (make sure to replace 'your_project_name' with your actual project name)
artifact = run.use_artifact('your_project_name/trainval_data.csv:latest', type='dataset')

# Download the artifact's file to a local path
artifact_dir = artifact.download()
This code snippet assumes you have W&B set up and have previously logged trainval_data.csv as an artifact in your project. Make sure to replace 'your_project_name' and 'your_wandb_username' with your actual W&B project name and username. The artifact.download() method downloads the artifact to your local environment, and you can then use it as needed in your pipeline or analyses.

User
am i supposed to run my data_split 
GPT
Yes, you should run the data_split step in your machine learning pipeline to divide your dataset into training and validation (and possibly test) sets. This step is crucial for preparing your data for the training phase and for evaluating the performance of your model on unseen data.

Here's a general outline of what you need to do to run the data_split step, assuming you're using MLflow and possibly integrating with Weights & Biases (W&B) for tracking experiments and artifacts:

Check the data_split Script: Ensure that the script or component responsible for the data splitting (train_val_test_split in your case) is correctly implemented. It should take your dataset as input and produce two or more outputs: training set, validation set, and optionally a test set.

Define Parameters in MLproject: Make sure that the MLproject file in your train_val_test_split directory correctly defines the data split step, including all necessary parameters such as input, test_size, random_seed, and stratify_by.

Update main.py: In your main pipeline script (main.py), add a segment that calls the data_split step using mlflow.run(). You'll need to specify the path to the train_val_test_split component, the entry point (usually 'main'), and any parameters required by the step.

Run the Pipeline: Execute your pipeline, ensuring that the data_split step is included in the steps to be run. If you're running the pipeline from the command line, this might look something like mlflow run . -P steps=data_split. If your pipeline is set up to run all steps by default, simply running the pipeline might suffice.

Check Artifacts: After running the pipeline, check your W&B project (if you're using W&B) to see if the trainval_data.csv and test_data.csv artifacts have been logged. If everything is set up correctly, these artifacts should be available under the "Artifacts" section of your W&B project dashboard.

Remember, the exact commands and steps might vary based on how your project is structured and the specific technologies and platforms you're using. If you're following a tutorial or documentation, refer to those resources for more detailed instructions tailored to your setup.

User
Step 5: Model Selection and Test
You've trained the model with different hyperparameters. Now you can select the best model and test the model with the test dataset. In this section, you will work in W&B to choose the best model and test the model by adding the test step components/test_regression_model to the ML pipeline.

Select the best model
Go to W&B and select the best performing model.

Look for your best model within W&B
Selecting the best model

HINT: you should switch to the Table view (second icon on the left), then click on the upper right on "columns", remove all selected columns by clicking on "Hide all", then click on the left list on "ID", "Job Type", "max_depth", "n_estimators", "mae" and "r2". Click on "Close". Now in the table view you can click on the "mae" column on the three little dots, then select "Sort asc". This will sort the runs by ascending Mean Absolute Error (best result at the top).

When you have found the best job, click on its name, then go to its artifacts and select the "model_export" output artifact. You can now add a prod tag to it to mark it as "production ready".

Test the model
Use the provided step components/test_regression_model to test your production model against the test set.

Add the step to the pipeline
Add this step in the main.py under test_regression_model step. As usual, you can see the parameters in the components/test_regression_model/MLproject file.

Use the artifact random_forest_export:prod for the parameter mlflow_model and the test artifact test_data.csv:latest as test_artifact.

Hint: the implementation of this step is similar to the data_split step.

Run the step
This step is NOT run by default when you run the pipeline. In fact, it needs the manual step of promoting a model to prod before it can complete successfully. That is what you have just done earlier. Now use the command line below to run the model:

> mlflow run . -P steps=test_regression_model
GPT
Talked to pluginlab.askthecode.dsomok.online
Step 5: Model Selection and Test
After training your model with different hyperparameters, the next step is to select the best performing model based on the criteria such as Mean Absolute Error (MAE) or R-squared (R2) and test it against your test dataset. Here's how you can proceed:

Select the Best Model in Weights & Biases (W&B)
Navigate to W&B Dashboard: Log in to your W&B account and access your project's dashboard.

Switch to Table View: On the left panel, click on the second icon to switch to the Table view for easier navigation and sorting of your runs.

Configure Columns: In the upper right corner, click on "columns", choose "Hide all", and then select only "ID", "Job Type", "max_depth", "n_estimators", "mae", and "r2". This will simplify the table and focus on the essential metrics.

Sort Runs by MAE: Click on the "mae" column to sort the runs by Mean Absolute Error in ascending order. The run with the lowest MAE is typically considered the best model.

Tag the Best Model: Click on the best run to view its details. Navigate to the "Artifacts" section and locate the "model_export" output artifact. Add a "prod" tag to it to mark it as production-ready.

Test the Model
To test the selected model against the test dataset, use the test_regression_model step provided in the components/test_regression_model.

Add the Test Step to the Pipeline
In main.py, under the test_regression_model step, add the following segment:

python
Copy code
if "test_regression_model" in active_steps:
    _ = mlflow.run(
        f"{config['main']['components_repository']}/test_regression_model",
        'main',
        parameters={
            "mlflow_model": "random_forest_export:prod",
            "test_dataset": "test_data.csv:latest"
        }
    )
This will configure the pipeline to use the production-ready model (random_forest_export:prod) and the latest test dataset (test_data.csv:latest) for testing.

Run the Test Step
To run this specific step, execute the following command in your terminal:

bash
Copy code
mlflow run . -P steps=test_regression_model
This command initiates the test_regression_model step, which evaluates the selected model against the test dataset. The results, including performance metrics like MAE and R2 for the test set, will be logged to your W&B project.

By following these steps, you complete the model selection and testing phase of your ML pipeline, ensuring that the model you deploy is the best performing one based on your criteria.

User
Step 6: Pipeline Release and Updates
Congratulations! You have tested the model and completed the ML pipeline. Now is the time to release the pipeline. In this section, you will first visualize the pipeline in W&B and then release the pipeline in your Github Repo. Next, you will use the released pipeline to train the model on a new dataset. Let's dive in.

Visualize the pipeline
You can now go to W&B, and go to the Artifacts section. Select the model export artifact then click on the Lineage tab. You will see something like this:

Visualization of the pipeline
Release the pipeline
First, copy the best hyperparameters you found into the config.yaml so they become the default values. Then, push your changes to the forked project repository on your Github repo. Finally, go to the repository on GitHub and make a release.

If you need a refresher, here are some instructions(opens in a new tab) on how to release on GitHub.

Call the release 1.0.0:

Tagging release 1.0.0 on Github, along with a release message. This example uses "Train a random forest model to the NYC Airbnb dataset" as its message.
Tagging release 1.0.0 on Github

If you find problems in the release, fix them and then make a new release like 1.0.1, 1.0.2 and so on.

Train the model on a new data sample
Let's now test that we can run the release using mlflow without any other pre-requisite. We will train the model on a new sample of data that our company received (sample2.csv):

(be ready for a surprise, keep reading even if the command fails)

> mlflow run https://github.com/[your github username]/Project-Build-an-ML-Pipeline-Starter.git \
             -v [the version you want to use, like 1.0.0] \
             -P hydra_options="etl.sample='sample2.csv'"
But, wait! It failed! The test test_proper_boundaries failed. Apparently, there is one point that is outside of the boundaries. This is an example of a "successful failure", i.e., a test that did its job and caught an unexpected event in the pipeline (in this case, in the data).

You can fix this by adding these two lines into the src/basic_cleaning/run.py just before the # Save the cleaned data section. You should see a TODO indication.

idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
This will drop rows in the dataset that are not in the proper geolocation.

Then commit your change, make a new release (for example 1.0.1), and rerun the pipeline (of course, you need to use -v 1.0.1 when calling mlflow this time). Now the run should succeed, and you have trained your new model on the new data.
GPT
Step 6: Pipeline Release and Updates
You've successfully completed the ML pipeline development and testing. Now, it's time to release the pipeline and make it ready for production use or further development. Here's how you can proceed:

Visualize the Pipeline in W&B
Access W&B Dashboard: Log into your Weights & Biases account and navigate to your project.
Explore Artifacts: Go to the "Artifacts" section and find the "model export" artifact.
View Lineage: Click on the "Lineage" tab to visualize the pipeline, showing the flow from data ingestion to model training and evaluation.
Release the Pipeline
Update config.yaml: Incorporate the best hyperparameters discovered during the experimentation phase into the config.yaml. This ensures that these parameters are used as defaults for future runs.

Push Changes to GitHub: Commit and push your changes to your forked project repository on GitHub. Include the updated config.yaml and any other modifications made during the pipeline development.

Create a Release on GitHub:

Navigate to your repository on GitHub.
Go to the "Releases" section and click "Draft a new release".
Use a tag like 1.0.0 for your first release, and add a title and description that summarize the pipeline's functionality.
Publish the release.
If you encounter issues, address them, commit the fixes, and create a new release (e.g., 1.0.1, 1.0.2, etc.).

Train the Model on New Data
With the pipeline released, you can now use it to train models on new data samples:

bash
Copy code
mlflow run https://github.com/[your_github_username]/Project-Build-an-ML-Pipeline-Starter.git \
             -v [version] \
             -P hydra_options="etl.sample='sample2.csv'"
Replace [your_github_username] with your actual GitHub username and [version] with the release version you want to use (e.g., 1.0.0).

Handling Failures
If the run fails due to data issues (like points outside of expected geographical boundaries), follow these steps to address it:

Update Data Cleaning Logic: In src/basic_cleaning/run.py, add a filtering step to remove rows with geolocation data outside the desired boundaries:

python
Copy code
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
Place this just before the section where the cleaned data is saved.

Commit and Push Changes: After making the necessary corrections, commit the changes and push them to your GitHub repository.

Create a New Release: Draft a new release on GitHub with an incremented version number (e.g., 1.0.1) and publish it.

Rerun the Pipeline: Execute the mlflow run command again with the updated version number to train the model on the new data sample.

By following these steps, you ensure that your ML pipeline is robust, adaptable to new data, and ready for production or further development iterations.

User
help me with Push Changes to GitHub
GPT
Pushing changes to your GitHub repository involves a series of steps that you can perform using Git from your command line or terminal. Here's a step-by-step guide to help you push your changes:

Step 1: Open your Terminal or Command Prompt
Navigate to the project directory where you have made the changes. You can do this using the cd command followed by the path to your directory. For example:

bash
Copy code
cd path/to/your/project-directory
Step 2: Check the Status of Your Changes
Before committing your changes, it's a good practice to check which files have been modified. You can do this with the git status command:

bash
Copy code
git status
This command will list all the files that have been changed or are new and not yet tracked by Git.

Step 3: Stage Your Changes
To prepare your changes for a commit, you need to stage them using the git add command. If you want to stage all modified files, you can use:

bash
Copy code
git add .
Or, if you prefer to stage specific files, you can specify them one by one:

bash
Copy code
git add file1.py file2.py
Step 4: Commit Your Changes
Once your changes are staged, you can commit them to your local repository with a message describing what you have done. Use the git commit command followed by the -m flag and your commit message in quotes:

bash
Copy code
git commit -m "Your descriptive commit message here"
Make sure your commit message is clear and descriptive of the changes you have made.

Step 5: Push Your Changes to GitHub
Finally, you can push your committed changes to your remote GitHub repository. If you are working on the default branch (often named main or master), you can use:

bash
Copy code
git push origin main
Or, if you are working on a different branch, replace main with the name of your branch:

bash
Copy code
git push origin your-branch-name
Additional Tips
If you haven't set up your remote repository yet, you'll need to link your local repository to your GitHub repo using the git remote add origin command followed by your repository's URL:

bash
Copy code
git remote add origin https://github.com/your-username/your-repo-name.git
If you're pushing to GitHub for the first time or if your credentials are not saved, you might be prompted to log in to GitHub.

If you encounter any errors or need help with specific Git commands, you can use git help followed by the command name for more information, or simply git help for an overview of available commands.

By following these steps, you should be able to successfully push your changes to your GitHub repository.

User
gimme a good title/descriptio nfor my release short and sweet
GPT
For your release title and description, you can capture the essence of your project's functionality and the significance of this particular release. Here's a suggestion:

Title:
"NYC Airbnb Price Prediction Model v1.0.0"

Description:
"Introducing the first stable release of our NYC Airbnb Price Prediction Model. This version features a meticulously tuned Random Forest algorithm, capable of forecasting Airbnb rental prices in New York City with improved accuracy. Enhanced data cleaning and validation steps ensure robust performance, making this release ideal for stakeholders looking to optimize pricing strategies. Dive into a more predictive future with v1.0.0."

This title and description succinctly communicate the purpose of the project and highlight the improvements and capabilities of this release, setting clear expectations for potential users.

User
always remember my username is dariusstrongman:

mlflow run https://github.com/[your_github_username]/Project-Build-an-ML-Pipeline-Starter.git \
             -v [version] \
             -P hydra_options="etl.sample='sample2.csv'"
GPT
Given your username is dariusstrongman, here's how you should run the command to execute your MLflow project with the specified version and options:

bash
Copy code
mlflow run https://github.com/dariusstrongman/Project-Build-an-ML-Pipeline-Starter.git \
             -v [version] \
             -P hydra_options="etl.sample='sample2.csv'"
Replace [version] with the actual version tag you want to use, for example, 1.0.0 or 1.0.1. This command will run the specified version of your MLflow project and use sample2.csv as the data sample for the ETL process.

User
and ill use version 1.0.0
GPT
Given that information, here's the command you should run to execute your MLflow project version 1.0.0:

bash
Copy code
mlflow run https://github.com/dariusstrongman/Project-Build-an-ML-Pipeline-Starter.git \
             -v 1.0.0 \
             -P hydra_options="etl.sample='sample2.csv'"
This command will run version 1.0.0 of your MLflow project, applying the etl.sample option to use sample2.csv as the dataset for your ETL process.

User
add to my read me for github, here is my public link: https://wandb.ai/dariuswork/nyc_airbnb?workspace=user-dariusstroman
GPT
To add a section about your project's integration with Weights & Biases (W&B) to your GitHub README, you could include the following content:

Weights & Biases Integration
This project is integrated with Weights & Biases (W&B), an ML experiment tracking tool, to monitor and visualize the machine learning pipeline's performance. The W&B dashboard provides insights into the model's training metrics, hyperparameters, and outputs, ensuring a transparent and reproducible ML workflow.

Explore the W&B Dashboard
To view detailed reports, visualizations, and comparisons of the model's performance, visit the project's W&B dashboard:

🔗 [NYC Airbnb Price Prediction Dashboard](https://wandb.ai/dariuswork/nyc_airbnb?workspace=user-dariusstroman)

The dashboard includes:

Training and validation loss curves
Hyperparameter tuning results
Model prediction visualizations
Artifacts lineage and versioning
Feel free to explore the various tabs and features within the W&B dashboard to gain deeper insights into the model's performance and the data it processes.

This README section provides an overview of the project's integration with W&B, directs readers to the W&B dashboard for detailed insights, and highlights key features and metrics available on the dashboard. Adjust the content as needed to match your project's specifics and the information available in your W&B dashboard.

User
make it seem like a freshman i ncollege wrote it, nothing too crazy
GPT
Sure! Here's a more simplified version for your GitHub README:

We're Using Weights & Biases!
Hey there! In this project, we're using this super cool tool called Weights & Biases to keep track of all our ML experiments. It's like having a super detailed diary for our machine learning model, where we can see how well it's learning and make it smarter over time!

Check Out Our Project Dashboard
Want to see how our project is doing? Click on the link below to check out our dashboard on Weights & Biases. It's where we keep all our project's progress and results:

🔗 Check out our NYC Airbnb Dashboard here!

On the dashboard, you'll find stuff like:

How well our model is learning over time
The settings we played with to make our model better
Cool charts that show what our model is predicting
A history of all the changes we made to our model
Feel free to poke around and explore the dashboard to see what we've been up to!



# Build an ML Pipeline for Short-Term Rental Prices in NYC
You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Table of contents

- [Preliminary steps](#preliminary-steps)
  * [Fork the Starter Kit](#fork-the-starter-kit)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Pre-existing components](#pre-existing-components)

## Preliminary steps
### Fork the Starter kit
Go to [https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter)
and click on `Fork` in the upper right corner. This will create a fork in your Github account, i.e., a copy of the
repository that is under your control. Now clone the repository locally so you can start working on it:

```
git clone https://github.com/[your github username]/Project-Build-an-ML-Pipeline-Starter.git
```

and go into the repository:

```
cd Project-Build-an-ML-Pipeline-Starter
```
Commit and push to the repository often while you make progress towards the solution. Remember 
to add meaningful commit messages.

### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yaml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```


### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 
Open this file and get familiar with its content. Remember: this file is only read by the ``main.py`` script 
(i.e., the pipeline) and its content is
available with the ``go`` function in ``main.py`` as the ``config`` dictionary. For example,
the name of the project is contained in the ``project_name`` key under the ``main`` section in
the configuration file. It can be accessed from the ``go`` function as 
``config["main"]["project_name"]``.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be 
accessed from the configuration file.

### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

### Pre-existing components
In order to simulate a real-world situation, we are providing you with some pre-implemented
re-usable components. While you have a copy in your fork, you will be using them from the original
repository by accessing them through their GitHub link, like:

```python
_ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
```
where `config['main']['components_repository']` is set to 
[https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/tree/main/components](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/tree/main/components).
You can see the parameters that they require by looking into their `MLproject` file:

- `get_data`: downloads the data. [MLproject](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/blob/main/components/get_data/MLproject)
- `train_val_test_split`: segrgate the data (splits the data) [MLproject](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/blob/main/components/train_val_test_split/MLproject)

## In case of errors
When you make an error writing your `conda.yml` file, you might end up with an environment for the pipeline or one
of the components that is corrupted. Most of the time `mlflow` realizes that and creates a new one every time you try
to fix the problem. However, sometimes this does not happen, especially if the problem was in the `pip` dependencies.
In that case, you might want to clean up all conda environments created by `mlflow` and try again. In order to do so,
you can get a list of the environments you are about to remove by executing:

```
> conda info --envs | grep mlflow | cut -f1 -d" "
```

If you are ok with that list, execute this command to clean them up:

**_NOTE_**: this will remove *ALL* the environments with a name starting with `mlflow`. Use at your own risk

```
> for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

This will iterate over all the environments created by `mlflow` and remove them.


## License

[License](LICENSE.txt)
