# AISA Study Project ML Deployment Options

## Abstract

In this study, we conducted a comparative analysis of deployment options for ML components. As an initial step, we identified existing deployment options, such as AWS SageMaker, Azure ML, BentoML, etc. We also created a list of criteria for comparison, such as their supported input frameworks, model explainability and monitoring during deployment. Furthermore, we compared the options side by side by reviewing their documentation and put together the list of data sources. Our comparative analysis is presented in a table and the data sources are available in our GitLab repository. Finally, some comparison criteria addressing practical metrics such as deployment time and ease of use will require hands-on tests, this can be the topic of future research. Our study assists practitioners in making decisions regarding the choice of deployment option based on their needs.

## Table of Contents

* [Introduction](./data-sources.md#Introduction)
* [Methods](#Methods)
* [Results](#Results)
  * [Deployment Options](#deployment-options)
  * [Criteria](#criteria)
* [Conclusion](#conclusion)
  * [Comparative Analysis Table](#comparative-analysis-table)
  * [Table Legend](#table-legend)
  * [Analysis](#analysis)
* [Future Work](#future-work)
* [License](#license)
* [Project status](#project-status)

## Introduction

In recent years, there has been a growing interest in deploying “smart systems”, which areable to emulate human behavior on specific tasks [a]. Typically, these systems employ artificialintelligence(AI) techniques, frequently machinelearning(ML), to achieve their human-like behavior. When deploying the system, we must also deploy the ML model as part of the system. While implementing a bespoke solution is always an option, recent years have also seen a rise in the number of off-the-shelf deployment options [b], e.g., TensorFlow Serving or MLflow (see [website](#deploymentoptions)). This growing competition could overwhelm practitioner's looking for a solution to deploy their ML models, for example, in the context of a make-or-buy decision. A comparative analysis ofoff-the-shelf deployment options across well-reasoned criteria would help them gain clarity about their needs and identify suitable options that fulfill these needs.

## Method

As the basis for the comparison, a very lightweight, non-systematic literature survey approach [c–e] identified criteria and off-the-shelf deployment options. This literature survey included grey literature since many deployment options have not been part of the academic discussion before. Grey literature also identified prior work on comparing deployment options, which served as the foundation for the rest of the study. Based on the literature survey, a catalog of criteria was synthesized. The catalog was then be employed to analyze the documentation of each deployment option and synthesizea comparison of the identified options [f,g]. The catalog of criteria and the results of the comparison was documented in a scientificreport and presented as a poster.

## Result

Check [data-sources.md](./data-sources.md) for the sources of the data presented in our findings.

TODO: create a QR code, add it to the poster and link it to this repo.

### Deployment Options

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

### Criteria

* [Supported ML frameworks](#supported-ml-frameworks)
* [Model Metrics Monitoring](#model-metrics-monitoring)
* [Anomaly detection](#anomaly-detection)
* [Model explainability](#model-explainability)
* [CI/CD Integration](#continuous-integration-and-continuous-delivery)
* [Popularity](#popularity)
* [Cost plan](#cost-plan)
* [Compatibility with Docker/Docker support](#compatibility-with-dockerdocker-support)
* [Offline batch processing/Request batching](#offline-batch-processing/request-batching)

## Conclusion

### Comparative Analysis Table

| Criteria                     | AWS SageMaker                                        | Azure ML                                             | BentoML                                              | Kubeflow                                             | MLflow                                               | RayServe                                             | Seldon Core                                          | TensorFlow Serving                    | TorchServe                            | Vertex AI                                            |
| ---------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| [Supported ML frameworks](#supported-ml-frameworks)† | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | only<br />TensorFlow models           | only Pytorch models                   | Pytorch,<br />Tensorflow, <br />Scikit Learn, others |
| [Model Metrics Monitoring](#model-metrics-monitoring)     | ✔                                                   | ✔                                                   | ✔                                                   | ◕                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |
| [Anomaly detection](#anomaly-detection)            | ✔                                                   | ✔                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ❓                                                   | ✔                                                   | ❓                                    | ❓                                    | ✔                                                   |
| [Model explainability](#model-explainability)           | ✔                                                   | ✔                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ✔                                                   | ❓                                    | ◕                                    | ✔                                                   |
| [CI/CD Integration](#continuous-integration-and-continuous-delivery)                        | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ◕                                                   | ❓                                                   | ◕                                                   | ✔                                    | ❓                                    | ❓                                                   |
| [Popularity](#popularity)                   | ❓                                                   | ❓                                                   | 6.2k                                                 | 13.4k                                                | 16.3k                                                | 29.6k                                                | 4.1k                                                 | 6k                                    | 3.8k                                  | ❓                                                   |
| [Cost plan](#cost-plan)                    | Free Tier +<br />different plans based on usage      | purchase plan<br />offered after consultation        | Open-source*<br />(Apache License 2.0)               | Open-source*<br />(Apache License 2.0)               | Open-source<br />(Apache License 2.0)                | Open-source<br />(Apache License 2.0)                | Open-source*<br />(Business Source License 1.1)      | Open-source<br />(Apache License 2.0) | Open-source<br />(Apache License 2.0) | Pay-as-you-go                                        |
| [Docker support](#compatibility-with-dockerdocker-support)               | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |
| [Offline batch processing/Request batching](#offline-batch-processing/request-batching)     | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ❌                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |

†: Most of the deployment frameworks support models created in various ML frameworks, the names of only the most popular ones are listed in the respective cells due to space constraints. See [data source](TODO: add path) file for details.

*: This indicates that special conditions apply to this entry. Please refer to notes in the [data source file](./data-sources.md) for specific details.

### Table legend

| **Symbol** | **Description**                                                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✔               | It is specified clearly on its official documentation that the feature is made available by the framework and not external integration or libraries are required for this purpose. |
| ◕               | It is specified clearly on its official documentation that the feature is made available with the help of external integration or libraries.                                       |
| ❓               | It is not specified clearly on its official documentation about the availability of this feature.                                                                                  |
| ❌               | It is specified that the feature is not available with the framework.                                                                                                              |

### Analysis

As platform-exclusive, such as TensorFlow Serving, which specializes in deployment services for TensorFlow models, and Torch Serve, which only supports pytorch models, their documentation does not mention that they have monitoring and explanation, anomaly detection.

Meanwhile, compared to platforms with their own large machine learning ecosystems such as AWS SageMaker belonging to Amazon and Azure ML belonging to Microsoft, and Vertex AI belonging to Google, some open source deployment platforms such as BentoML, Kubeflow,Mlflow, RayServe are generally compatible with most of the ML model types.

Some open source deployment platforms such as BentoML, Kubeflow, Mlflow, RayServe are generally compatible with most of the ML model input types, but have limited functionality such as model monitoring and anomaly detection and model explanation and offline batch processing. However, they tend to collaborate with other open source platforms such as Captum. grafana and alibi detect, which is a huge advantage for them as an open source platform.

From a business point of view, there is no doubt that reliable and well established features and ecosystems are the strengths of AWS SageMaker, Vertex AI, and Azure ML, and at the same time, this corresponds to the corresponding expenses, and they have different payment rates, which basically depends on the user's purpose and scale of use.Additionally although most of the open source platforms have an open source While most of the open source platforms have an open source, free nature, Selfdon Core has a Business Source License 1.1 that restricts the use of some of them. Finally, the most popular of the open source projects is RayServe, which we believe is related to its "scalable model serving library for building online inference APIs". As an existing issue, it has been documented that MLflow does not have the capability for Offline batch processing, and that developing this capability could be a future research direction.

## Future Work

* Conducting hands-on experiments and deployment tests for each machine learning deployment framework to assess more criteria. More specifically, **Deployment Time Analysis**, **Cost Evaluation**, **Cross-Platform Compatibility and Interoperability**, **Scalability and Performance Testing, User Experience** and **Ease of Use** should be considered.
* **Extend the comparative analysis** to include ML frameworks for other stages of the MLops process, such as data preparation, model design and model training.
* **Refine the analysis** for domain specific use cases, compare and analyze the strengths and weaknesses of these framework in task specific models i.e., computer vision models, LLMs, LSTMs, etc. This is because different frameworks are optimized for different tasks.
* **Market analysis** by comparing their popularity and pricing plan in different aspects such as user-group, security, low-latency use case, etc.
* **Development** of Offline Batch processing for MLflow.

## References

[a] Saleema Amershi et al. “Software Engineering for Machine Learning: A Case Study”. In:2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineeringin Practice (ICSE-SEIP). Montreal, QC, Canada (May 25–31, 2019). Montreal, QC, Canada:IEEE, May 2019, pp. 291–300. ISBN: 978-1-7281-1761-4. DOI:10.1109/ICSE-SEIP.2019.00042.

[b] Sherif Akoush et al. “Desiderata for next generation of ML model serving”. In: (Oct. 26,2022). DOI:10.48550/ARXIV.2210.14665. arXiv:2210.14665 [cs.LG].

[c] Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. “Rapid Reviews in Software Engineer-ing”. In: Springer International Publishing, 2020, pp. 357–384. DOI:10.1007/978-3-030-32489-6_13.

[d] Hannah Snyder. “Literature review as a research methodology: An overview and guide-lines”. In:Journal of Business Research104 (Nov. 2019), pp. 333–339. DOI:10 . 1016 / j .jbusres.2019.07.039.

[e] Richard J. Torraco. “Writing Integrative Literature Reviews: Guidelines and Examples”.In:Human Resource Development Review4.3 (Sept. 2005), pp. 356–367. DOI:10.1177/1534484305278283.

[f] Philipp Mayring.Qualitative content analysis: theoretical foundation, basic procedures andsoftware solution. Tech. rep. Klagenfurt, 2014.
