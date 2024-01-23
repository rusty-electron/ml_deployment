# AISA Study Project ML Deployment Options

QR code for varifications keep in gitlab with open sourced access.

## Deployment Options

* [AWS Sagemaker](https://aws.amazon.com/sagemaker/)

  * not much customization available
  * can be used only with AWS ecosystem
* [Azure ML](https://azure.microsoft.com/en-us/free/machine-learning/)

  * [see more info here](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model?view=azureml-api-2)
* [BentoML](https://docs.bentoml.com/en/latest/)
* [KubeFlow](https://www.kubeflow.org/)

  * integration with Kubernetes
  * Open-source
  * Serving using KServe
* [MLflow](https://mlflow.org/)

  * Open-source
* [RayServe](https://docs.ray.io/en/latest/serve/index.html)
* [Seldon Core](https://www.seldon.io/solutions/core-plus)
* [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)
* [TorchServe](https://pytorch.org/serve/)

  * work only with Pytorch models
  * simple and lightweight
* [Vertex AI](https://cloud.google.com/vertex-ai/docs) (Google Cloud)

## Criteria

* Number of supported input plattforms: it can deploy any common model built with popular ML training frameworks. e.g., onnx, tensorflow, keras, etc. [2]
  * Kubeflow: PyTorch, TensorFlow, Hugging Face, Jupyter, scikit-learn, HOROVOD, dmlcXGBoost (we may convert this to formats)
* Outlier detection: it is the process of identifying data points that deviate to a large extent from the overall pattern of the data. This is crucial in machine learning (ML) deployment, as it helps identify anomalies in the input data that could negatively impact model performance. [3]
  * Kubeflow: via Alibi-detect from Seldon Core
    * https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md
    * 
  * Seldon Core: Yes
* Data drift detection: data drift occurs when changes are observed in joint distributionp(X, y), whereXis the model input andyis the model output. Detection of data drift is a growing concern for teams that maintain ML models in production [8].
  * Kubeflow: via Alibi-detect from Seldon Core
    * https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md
* Ability to monitor model metrics: As it is not possible to check the output labels for live
  data, the statistics of input data and output predictions are monitored as a proxy for model performance. [3]
  * Kubeflow: via Alibi-detect from Seldon Core
    * https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md
* Model explainability: Explanation algorithms should be available in order to provide insights into the decision process of the model. Explainability in the outputs of a deployed model help in building trust in the ML system [3].
* Continuous integration and continuous delivery: CI/CD is a powerful tool that can help ML teams develop and deploy models faster and more efficiently [8].
* Deployment Focus: Which kind of ML models does the option support to deploy, e.g. Only Machine Learning models or Web applications possible.[4] PG: being able to deploy web applications too might not be a priority for ML engineers so this criterion is a low priority one but can be included.
* Cost plan: How much does it cost e.g. monthly?[4] This is a good criterion :). We can find this data from their pricing page of the respective websites. This can also include feature like cost monitoring.
* Compatibility with Docker/Docker support: Docker containers encapsulate all the dependencies, libraries, and configurations needed for an application. It also allows for the creation of reproducible environments. This means that the same Docker container can be run on different machines or cloud services, ensuring that the ML model’s behavior remains consistent across various deployment scenarios. In many cases, platforms can offer pre-built docker images for common use-cases. [6]
* Offline batch processing/Request batching: it refers to performing predictions on a batch of data in a scheduled, non-interactive, and often offline manner. Some models are not deployed for real-time applications, they can utilize the parallelizing capabilities of hardware accelerators to wait for a batch of requests to accumulate and then complete them together. [2]


## Conclusion

| Deployment Option     | Supported Platforms    | Outlier Detection | Data Drift Detection | Model Monitoring | Model Explainability | CI/CD | Deployment Focus | Cost Plan | Docker Support | Batch Processing |
| --------------------- | ---------------------- | ----------------- | -------------------- | ---------------- | -------------------- | ----- | ---------------- | --------- | -------------- | ---------------- |
| AWS SageMaker         | Broad (AWS Ecosystem)  | ?                 | ?                    | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| *Azure ML*           | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| BentoML               | Broad                  | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free/Paid | ✔             | ✔               |
| *KubeFlow*           | Broad (via Converters) | ✔ (Seldon Core)  | ✔ (Seldon Core)     | ✔ (Seldon Core) | ✔ (Seldon Core)     | ✔    | ML Models        | Free      | ✔             | ✔               |
| MLflow                | Broad                  | ?                 | ?                    | ✔               | ✔                   | ✔    | ML Models        | Free      | ✔             | ✔               |
| *RayServe*           | Broad                  | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free/Paid | ✔             | ✔               |
| Seldon Core           | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| *TensorFlow Serving* | TensorFlow, Keras      | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free      | ✔             | ✔               |
| TorchServe            | PyTorch Only           | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free      | ✔             | ✔               |
| *Vertex AI*          | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models & Apps | Paid      | ✔             | ✔               |

## Future Work

## Test and Deploy

Do we need to test any code?

## License

Do we need License for this project?

## Project status

Begins at 2023-12-04
