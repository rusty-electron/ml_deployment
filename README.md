# AISA Study Project ML Deployment Options

TODO: add a short description of the project or abstract

#### Check [data-sources.md](./data-sources.md) for the sources of the data presented in our findings.

TODO: create a QR code, add it to the poster and link it to this repo.

TODO: add a TOC

## Deployment Options

* [AWS Sagemaker](https://aws.amazon.com/sagemaker/)
* [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model?view=azureml-api-2)
* [BentoML](https://docs.bentoml.com/en/latest/)
* [KubeFlow](https://www.kubeflow.org/)
* [MLflow](https://mlflow.org/)
* [RayServe](https://docs.ray.io/en/latest/serve/index.html)
* [Seldon Core](https://www.seldon.io/solutions/core-plus)
* [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)
* [TorchServe](https://pytorch.org/serve/)
* [Vertex AI (Google Cloud)](https://cloud.google.com/vertex-ai/docs)

## Criteria

* **Number of supported input model type**: it can deploy any common model built with popular ML training frameworks. e.g., onnx, tensorflow, keras, etc. [2]
* **Model Metrics Monitoring**: As it is not possible to check the output labels for live data, the statistics of input data and output predictions are monitored as a proxy for model performance. [3]
* **Anomaly detection**: Also known as **outlier detection**, it is the process of identifying data points that deviate to a large extent from the overall pattern of the data. This is crucial in machine learning (ML) deployment, as it helps identify anomalies in the input data that could negatively impact model performance. [3] Additionally, **data drift detection** is part of this criterion, data drift occurs when changes are observed in joint distribution $p(X, y)$, where $X$ is the model input and $y$ is the model output. Detection of data drift is a growing concern for teams that maintain ML models in production [8].
* **Model explainability** (might not be relevant for deployment tools): Explanation algorithms should be available in order to provide insights into the decision process of the model. Explainability in the outputs of a deployed model help in building trust in the ML system [3].
* **Continuous integration and continuous delivery**: CI/CD is a powerful tool that can help ML teams develop and deploy models faster and more efficiently [8].
* **Deployment Focus**: Which kind of ML models does the option support to deploy, e.g. Only Machine Learning models or Web applications possible.[4]
* **Cost plan**: How much does it cost e.g. monthly? [4]
* **Compatibility with Docker/Docker support**: Docker containers encapsulate all the dependencies, libraries, and configurations needed for an application. It also allows for the creation of reproducible environments. This means that the same Docker container can be run on different machines or cloud services, ensuring that the ML model’s behavior remains consistent across various deployment scenarios. In many cases, platforms can offer pre-built docker images for common use-cases. [6]
* **Offline batch processing/Request batching**: it refers to performing predictions on a batch of data in a scheduled, non-interactive, and often offline manner. Some models are not deployed for real-time applications, they can utilize the parallelizing capabilities of hardware accelerators to wait for a batch of requests to accumulate and then complete them together. [2]

## Conclusion

TODO: after the final version of the table, convert each row header into a link to the corresponding section.

### Comparative Analysis Table

| Criteria                     | AWS SageMaker                                        | Azure ML                                             | BentoML                                              | Kubeflow                                             | MLflow                                               | RayServe                                             | Seldon Core                                          | TensorFlow Serving                            | TorchServe                                    | Vertex AI                                 |
| ---------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------- | --------------------------------------------- | ----------------------------------------- |
| Supported Input Model Type† | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | only<br />TensorFlow models                   | only Pytorch models                           | Pytorch, Tensorflow, Scikit Learn, others |
| Model Metrics Monitoring     | ✔                                                   | ✔                                                   | ✔                                                   | ➖                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                            | ✔                                            | ✔                                        |
| Anomaly Detection            | ✔                                                   | ✔                                                   | ➖                                                   | ➖                                                   | ➖                                                   | ❓                                                   | ✔                                                   | ❓                                            | ❓                                            | ✔                                        |
| Model Explainability         | ✔                                                   | ✔                                                   | ➖                                                   | ➖                                                   | ➖                                                   | ➖                                                   | ✔                                                   | ❓                                            | ➖                                            | ✔                                        |
| CI/CD                        | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ➖                                                   | ❓                                                   | ➖                                                   | ✔                                            | ❓                                            | ❓                                        |
| Popularity                   | ❓                                                   | ❓                                                   | 6.2k                                                 | 13.4k                                                | 16.3k                                                | 29.6k                                                | 4.1k                                                 | 6k                                            | 3.8k                                          | ❓                                        |
| Cost Plan                    | Free Tier +<br />different plans based on usage      | purchase plan<br />offered after consultation        | Open-source*<br />(Apache License 2.0)               | Open-source*<br />(Apache License 2.0)               | Open-source<br />(Apache License 2.0)                | Open-source<br />(Apache License 2.0)                | Open-source*<br />(Business Source License 1.1)      | Open-source<br />(Apache License Version 2.0) | Open-source<br />(Apache License Version 2.0) | Pay-as-you-go                             |
| Docker Support               | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                            | ✔                                            | ✔                                        |
| Offline Batch Processing     | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ❌                                                   | ✔                                                   | ✔                                                   | ✔                                            | ✔                                            | ✔                                        |

†: Most of the deployment frameworks support models created in various ML frameworks, the names of only the most popular ones are listed in the respective cells due to space constraints. See [data source](TODO: add path) file for details.

*: This indicates that special conditions apply to this entry. Please refer to notes in the [data source file](./data-sources.md) for specific details.

| **Symbol** | **Description**                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| ✔               | it is specified clearly on its official website or docs that there is no need to incoorperate with external libraries or products |
| ➖               | it is specified clearly on its official website or docs that it can incoorperate with external libraries or products              |
| ❓               | it is not specified clearly on its official website or docs about the avalability of this feature                                 |
| ❌               | It is specified that the feature is not available with the framework                                                              |

### Disscussion and Strategy

TODO: either update or remove this section

* **Broad Support for ML Frameworks**: Azure ML, BentoML, Kubeflow, MLflow, and RayServe offer wide-ranging support for multiple machine learning frameworks, making them versatile choices for diverse model deployment needs. AWS SageMaker, while part of the AWS ecosystem, also supports various models but with less customization.
* **Model Metrics Monitoring**: All analyzed deployment options, including AWS SageMaker, Azure ML, BentoML, Kubeflow, MLflow, RayServe, Seldon Core, TensorFlow Serving, TorchServe, and Vertex AI, provide model metrics monitoring capabilities. This feature is crucial for maintaining and understanding model performance in production.
* **Anomaly Detection**: AWS SageMaker and Azure ML explicitly provide anomaly detection capabilities. BentoML offers this with some limitations, while the capability is uncertain or not a primary feature in Kubeflow, MLflow, RayServe, Seldon Core, TensorFlow Serving, TorchServe, and Vertex AI.
* **Model Explainability**: Azure ML stands out for offering model explainability, a feature that helps in understanding the decision-making process of the models. Other platforms either do not provide this feature or it's not clearly mentioned.
* **CI/CD Integration**: Azure ML and BentoML explicitly support continuous integration and continuous delivery, facilitating efficient development and deployment workflows.
* **Deployment Focus**: While most platforms are designed to deploy a variety of ML models, TensorFlow Serving is specialized for TensorFlow models, and TorchServe for PyTorch models. Vertex AI and Azure ML provide more general support for different types of ML models.
* **Cost**: Platforms like BentoML, Kubeflow, MLflow, RayServe, Seldon Core, TensorFlow Serving, and TorchServe are open-source and offer free usage, whereas AWS SageMaker and Vertex AI follow a pay-as-you-go pricing model. Azure ML operates on a consumption-based model.
* **Docker Support**: Most platforms offer Docker support, which is crucial for ensuring consistency across different deployment environments.
* **Offline Batch Processing**: Azure ML, BentoML, Kubeflow, RayServe, Seldon Core, TensorFlow Serving, and Vertex AI offer offline batch processing capabilities, an essential feature for models not requiring real-time prediction.

In summary, the choice of deployment platform depends on specific needs like framework support, monitoring capabilities, pricing, and deployment focus. Azure ML, AWS SageMaker, and Vertex AI, being cloud-based services, offer a comprehensive suite of features but at a cost. Open-source alternatives like BentoML, Kubeflow, and MLflow offer flexibility and no cost but might require more setup and integration effort. TensorFlow Serving and TorchServe are more specialized but highly efficient for their respective frameworks.

## Future Work

* Conducting hand-on experiments and deployment tests for each machine learning deployment option to assess more criteria. More specificlly, **Deployment Time Analysis**, **Cost Evaluation, **Cross-Platform Compatibility and Interoperability, **Scalability and Performance Testing, **User Experience and Ease of Use.********
* **Extend the comparative analysis** to include ML frameworks for other stages of the MLops process, such as data preparation, model design, and model training.
* **Refine the analysis** to each use case domain, compare and analyze the strengths and weaknesses of these options in each domain i.e. in training computer vision model, LLM etc. This can also extend to analysis of their most advanced libraries or features.
* **Market analysis** such like comparing their popularity and pricing plan in different aspects like user-group, security, unique selling points etc.

## License

TODO: add license after consulting Markus

## Project status

Started on 2023-12-04
