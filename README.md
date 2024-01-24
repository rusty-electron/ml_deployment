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
  * 
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

* Number of supported input model type: it can deploy any common model built with popular ML training frameworks. e.g., onnx, tensorflow, keras, etc. [2]

  * AWS SageMaker
  * Azure ML (https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-catalog?view=azureml-api-2#collections):
    * **Open source models curated by Azure AI**, 'Curated by Azure AI' and collections from partners such as **Meta, NVIDIA, Mistral AI** are all curated collections on the Catalog.
    * **Azure OpenAI models, exclusively available on Azure**
    * **Transformers models from the HuggingFace hub**
  * BentoML
  * Kubeflow:
    * model from **PyTorch, TensorFlow, Hugging Face, Jupyter, scikit-learn, HOROVOD, dmlcXGBoost**
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* **Anomaly detection**: This could include **outlier detection**, it is the process of identifying data points that deviate to a large extent from the overall pattern of the data. This is crucial in machine learning (ML) deployment, as it helps identify anomalies in the input data that could negatively impact model performance. [3] Additionally, **data drift detection** is part of this criteria, data drift occurs when changes are observed in joint distributionp(X, y), whereXis the model input andyis the model output. Detection of data drift is a growing concern for teams that maintain ML models in production [8].

  * AWS SageMaker
  * Azure ML: Yes (https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/introducing-azure-anomaly-detector-api/ba-p/490162)
    * Outlier Detection (Change Points detection): Yes (https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview)
    * Data Drift Detection: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?view=azureml-api-1&tabs=python)
  * BentoML
  * Kubeflow: Yes (via Alibi-detect from Seldon Core: https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md)
  * MLflow
  * RayServe
  * Seldon Core: Yes
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Model Metrics Monitoring: As it is not possible to check the output labels for live
  data, the statistics of input data and output predictions are monitored as a proxy for model performance. [3]

  * AWS SageMaker
  * Azure ML: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2)
  * BentoML
  * Kubeflow: Yes (via Alibi-detect from Seldon Core: https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md)
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Model explainability: Explanation algorithms should be available in order to provide insights into the decision process of the model. Explainability in the outputs of a deployed model help in building trust in the ML system [3].

  * AWS SageMaker
  * Azure ML: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-automl?view=azureml-api-1)
  * BentoML
  * Kubeflow:
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Continuous integration and continuous delivery: CI/CD is a powerful tool that can help ML teams develop and deploy models faster and more efficiently [8].

  * AWS SageMaker
  * Azure ML: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-event-grid?view=azureml-api-2)
  * BentoML
  * Kubeflow:
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Deployment Focus: Which kind of ML models does the option support to deploy, e.g. Only Machine Learning models or Web applications possible.[4] PG: being able to deploy web applications too might not be a priority for ML engineers so this criterion is a low priority one but can be included.

  * AWS SageMaker
  * Azure ML: any kind of ML models
  * BentoML
  * Kubeflow:
  * MLflow
  * RayServe: any kind of ML models ([Ray Serve: Scalable and Programmable Serving — Ray 2.9.1](https://docs.ray.io/en/latest/serve/index.html#ray-serve-scalable-and-programmable-serving))
  * Seldon Core
  * TensorFlow Serving: only TensorFlow models
  * TorchServe
  * Vertex AI: only ML models built with TensorFlow, XGBoost, Scikit-learn, or custom containers, as online or batch prediction services.
* Cost plan: How much does it cost e.g. monthly?[4] We can find this data from their pricing page of the respective websites. This can also include feature like cost monitoring.

  * AWS SageMaker: pay-as you-go
  * Azure ML: consumption-based
  * BentoML: open-source + cost on premise infrastructure
  * Kubeflow: open-source
  * MLflow: open-source
  * RayServe: open-source
  * Seldon Core: open-source
  * TensorFlow Serving: open-source
  * TorchServe: open-source
  * Vertex AI: pay-as you-go
* Compatibility with Docker/Docker support: Docker containers encapsulate all the dependencies, libraries, and configurations needed for an application. It also allows for the creation of reproducible environments. This means that the same Docker container can be run on different machines or cloud services, ensuring that the ML model’s behavior remains consistent across various deployment scenarios. In many cases, platforms can offer pre-built docker images for common use-cases. [6]

  * AWS SageMaker: Yes ([Use Docker containers to build models - Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html))
  * Azure ML: Yes ([microsoft-azureml - Official Image | Docker Hub](https://hub.docker.com/_/microsoft-azureml))
  * BentoML:
  * Kubeflow: Yes ([Container Images | Kubeflow](https://www.kubeflow.org/docs/components/notebooks/container-images/))
  * MLflow: Yes ([Official MLflow Docker Image — MLflow 2.9.2 documentation](https://mlflow.org/docs/latest/docker.html))
  * RayServe: Yes ([Trying to deploy ray with docker - Ray Libraries (Data, Train, Tune, Serve) / Ray Serve - Ray](https://discuss.ray.io/t/trying-to-deploy-ray-with-docker/766))
  * Seldon Core
  * TensorFlow Serving: Yes ([TensorFlow Serving with Docker  |  TFX](https://www.tensorflow.org/tfx/serving/docker))
  * TorchServe
  * Vertex AI: Yes ([Custom containers overview  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/training/containers-overview))
* Offline batch processing/Request batching: it refers to performing predictions on a batch of data in a scheduled, non-interactive, and often offline manner. Some models are not deployed for real-time applications, they can utilize the parallelizing capabilities of hardware accelerators to wait for a batch of requests to accumulate and then complete them together. [2]

  * AWS SageMaker
  * Azure ML: Yes ([Deploy machine learning models in production environments - Cloud Adoption Framework | Microsoft Learn](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/innovate/best-practices/ml-deployment-inference#architectural-considerations))
  * BentoML
  * Kubeflow: Yes (via Seldon Core: [Batch processing with Kubeflow Pipelines — seldon-core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_pipelines_batch.html))
  * MLflow
  * RayServe: Yes ([Dynamic Request Batching — Ray 2.9.1](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html))
  * Seldon Core: Yes ([Batch processing with Kubeflow Pipelines — seldon-core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_pipelines_batch.html))
  * TensorFlow Serving: Yes ([serving/tensorflow_serving/batching/README.md at master · tensorflow/serving · GitHub](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md#batch-scheduling-parameters-and-tuning))
  * TorchServe
  * Vertex AI ([Get batch predictions and explanations  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions))

## Conclusion

| Deployment Option      | Supported Platforms    | Outlier Detection | Data Drift Detection | Model Monitoring | Model Explainability | CI/CD | Deployment Focus | Cost Plan | Docker Support | Batch Processing |
| ---------------------- | ---------------------- | ----------------- | -------------------- | ---------------- | -------------------- | ----- | ---------------- | --------- | -------------- | ---------------- |
| AWS SageMaker          | Broad (AWS Ecosystem)  | ?                 | ?                    | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| *Azure ML*           | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| BentoML                | Broad                  | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free/Paid | ✔             | ✔               |
| *KubeFlow*           | Broad (via Converters) | ✔ (Seldon Core)  | ✔ (Seldon Core)     | ✔ (Seldon Core) | ✔ (Seldon Core)     | ✔    | ML Models        | Free      | ✔             | ✔               |
| MLflow                 | Broad                  | ?                 | ?                    | ✔               | ✔                   | ✔    | ML Models        | Free      | ✔             | ✔               |
| *RayServe*           | Broad                  | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free/Paid | ✔             | ✔               |
| Seldon Core            | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models        | Paid      | ✔             | ✔               |
| *TensorFlow Serving* | TensorFlow, Keras      | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free      | ✔             | ✔               |
| TorchServe             | PyTorch Only           | ?                 | ?                    | ✔               | ?                    | ✔    | ML Models        | Free      | ✔             | ✔               |
| *Vertex AI*          | Broad                  | ✔                | ✔                   | ✔               | ✔                   | ✔    | ML Models & Apps | Paid      | ✔             | ✔               |

## Future Work

## Test and Deploy

Do we need to test any code?

## License

Do we need License for this project?

## Project status

Begins at 2023-12-04
