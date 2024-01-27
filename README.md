# AISA Study Project ML Deployment Options

## Abstract

In this study, we conducted a comparative analysis of deployment options for ML components. We identified existing deployment options, such as AWS SageMaker, Azure ML, BentoML etc. We also created a list of criteria for comparison, such as their supported input model types, model explaination function and monitoring during deplyment. Furthermore, we compared them through reviewing of their documents and summarized the data sources. Our comparative analysis was presented in a table, the data sources were collected to one markdown file in GitLab repository. In conclusion, as platform-exclusive, such as TensorFlow Serving, which specializes in deployment services for TensorFlow models, and Torch Serve, which only supports pytorch models, their documentation does not mention that they have Monitoring and anomaly detection and model explanation. Meanwhile, compared to platforms with their own large machine learning ecosystems such as AWS SageMaker belonging to Amazon and Azure ML belonging to Microsoft, and Vertex AI belonging to Google, some open source deployment platforms such as BentoML, Kubeflow, Mlflow, RayServe are generally compatible with most of the ML model types. Some open source deployment platforms such as BentoML, Kubeflow, Mlflow, RayServe are generally compatible with most of the ML model input types, but have limited functionality such as model monitoring and anomaly detection and model explanation and offline batch processing. However, they tend to collaborate with other open source platforms such as Captum. grafana and alibi detect, which is a huge advantage for them as an open source platform. From a business point of view, there is no doubt that reliable and well established features and ecosystems are the strengths of AWS SageMaker, Vertex AI, and Azure ML, and at the same time, this corresponds to the corresponding expenses, and they have different payment rates, which basically depends on the user's purpose and scale of use.Additionally although most of the open source platforms have an open source While most of the open source platforms have an open source, free nature, Selfdon Core has a Business Source License 1.1 that restricts the use of some of them. Also, the most popular of the open source projects is RayServe, which we believe is related to its "scalable model serving library for building online inference APIs". Finally, some comparation criteria addressing practical metrics such as deployment time and ease of use are up to hand-on test, also, as an existing issue, it has been documented that MLflow does not have the capability for Offline batch processing, and that developing this capability could be a future research direction.

## Table of Contents

* [Introduction](#Introduction)
* [Objectives](#Objectives)
* [Methods](#Methods)
* [Results](#Results)
  * [Deployment Options](#deployment-options)
  * [Criteria](#criteria)
* [Conclusion](#conclusion)
  * [Comparative Analysis Table]()
  * Table Legend
  * Analysis
* [Future Work](#future-work)
* [License](#license)
* [Project status](#project-status)

## Introduction

In recent years, there has been a growing interest in deploying “smart systems”, which areable to emulate human behavior on specific tasks [a]. Typically, these systems employ artifi-cialintelligence(AI) techniques,frequentlymachinelearning(ML),toachievetheirhuman-likebehavior.When deploying the system, we must also deploy the ML model as part of the system. Whileimplementing a bespoke solution is always an option, recent years have also seen a rise inthe number of off-the-shelf deployment options [b], e.g., TensorFlow Serving or MLflow (see [website](#deploymentoptions)).ThisgrowingcompetitioncouldoverwhelmpractitionerslookingforasolutiontodeploytheirML models, for example, in the context of a make-or-buy decision. A comparative analysis ofoff-the-shelf deployment options across well-reasoned criteria would help them gain clarityabout their needs and identify suitable options that fulfill these needs. As platform-exclusive, such as TensorFlow Serving, which specializes in deployment services for TensorFlow models, and Torch Serve, which only supports pytorch models, their documentation does not mention that they have Monitoring and anomaly detection and model explanation. Meanwhile, it is worth mentioning that compared to platforms with their own large machine learning ecosystems such as Amazon's AWS SageMaker and Microsoft's Azure ML, and Google's Vertex AI, some open source deployment platforms such as BentoML, Kubeflow, Mlflow, RayServe, and TensorFlow Serving, and Torch Serve, which only support pytorch models, are not mentioned in their documents as having monitoring and anomaly detection and model explanation.

## Objectives

In this AISA study project, participants should conduct a comparative analysis of deploymentoptions for ML components. For this purpose, participants should identify existing deploy-ment options, such as tools or platforms. Participants should also create a list of criteria forcomparison, such as requirements on the deployed ML component or support for advanceddeployment strategies such as green-blue deployments.

## Method

As the basis for the comparison, a very lightweight, non-systematic literature survey approach [c–e] should identify criteria and off-the-shelf deployment options. This literature sur-vey should include grey literature since many deployment options may not have been part ofthe academic discussion before. Grey literature could also identify prior work on comparingdeployment options, which could serve as the foundation for the rest of the study.Basedontheliteraturesurvey, acatalogofcriteriashouldbesynthesized. Thecatalogshouldthen be employed to analyze the documentation of each deployment option and synthesizea comparison of the identified options [f,g].The catalog of criteria and the results of the comparison should be documented in a scientificreport and presented as a poster.

## Result

#### Check [data-sources.md](./data-sources.md) for the sources of the data presented in our findings.

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

TODO: update this section with latest description of the criteria.

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

| Criteria                     | AWS SageMaker                                        | Azure ML                                             | BentoML                                              | Kubeflow                                             | MLflow                                               | RayServe                                             | Seldon Core                                          | TensorFlow Serving                    | TorchServe                            | Vertex AI                                            |
| ---------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Supported Input Model Type† | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | Pytorch,<br />Tensorflow, <br />Scikit Learn, others | only<br />TensorFlow models           | only Pytorch models                   | Pytorch,<br />Tensorflow, <br />Scikit Learn, others |
| Model Metrics Monitoring     | ✔                                                   | ✔                                                   | ✔                                                   | ◕                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |
| Anomaly Detection            | ✔                                                   | ✔                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ❓                                                   | ✔                                                   | ❓                                    | ❓                                    | ✔                                                   |
| Model Explainability         | ✔                                                   | ✔                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ◕                                                   | ✔                                                   | ❓                                    | ◕                                    | ✔                                                   |
| CI/CD                        | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ◕                                                   | ❓                                                   | ◕                                                   | ✔                                    | ❓                                    | ❓                                                   |
| Popularity                   | ❓                                                   | ❓                                                   | 6.2k                                                 | 13.4k                                                | 16.3k                                                | 29.6k                                                | 4.1k                                                 | 6k                                    | 3.8k                                  | ❓                                                   |
| Cost Plan                    | Free Tier +<br />different plans based on usage      | purchase plan<br />offered after consultation        | Open-source*<br />(Apache License 2.0)               | Open-source*<br />(Apache License 2.0)               | Open-source<br />(Apache License 2.0)                | Open-source<br />(Apache License 2.0)                | Open-source*<br />(Business Source License 1.1)      | Open-source<br />(Apache License 2.0) | Open-source<br />(Apache License 2.0) | Pay-as-you-go                                        |
| Docker Support               | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |
| Offline Batch Processing     | ✔                                                   | ✔                                                   | ✔                                                   | ✔                                                   | ❌                                                   | ✔                                                   | ✔                                                   | ✔                                    | ✔                                    | ✔                                                   |

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

### A References for Introduction

[a] Saleema Amershi et al. “Software Engineering for Machine Learning: A Case Study”. In:2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineeringin Practice (ICSE-SEIP). Montreal, QC, Canada (May 25–31, 2019). Montreal, QC, Canada:IEEE, May 2019, pp. 291–300. ISBN: 978-1-7281-1761-4. DOI:10.1109/ICSE-SEIP.2019.00042.

[b] Sherif Akoush et al. “Desiderata for next generation of ML model serving”. In: (Oct. 26,2022). DOI:10.48550/ARXIV.2210.14665. arXiv:2210.14665 [cs.LG].

[c] Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. “Rapid Reviews in Software Engineer-ing”. In: Springer International Publishing, 2020, pp. 357–384. DOI:10.1007/978-3-030-32489-6_13.

[d] Hannah Snyder. “Literature review as a research methodology: An overview and guide-lines”. In:Journal of Business Research104 (Nov. 2019), pp. 333–339. DOI:10 . 1016 / j .jbusres.2019.07.039.

[e] Richard J. Torraco. “Writing Integrative Literature Reviews: Guidelines and Examples”.In:Human Resource Development Review4.3 (Sept. 2005), pp. 356–367. DOI:10.1177/1534484305278283.

[f] Philipp Mayring.Qualitative content analysis: theoretical foundation, basic procedures andsoftware solution. Tech. rep. Klagenfurt, 2014.

### B References for Criteria

[1] Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. Rapid Reviews in Software Engineering. In Michael Felderer and Guilherme Horta Travassos, editors,Contemporary Empirical Methods in Software Engineering, pages 357–384. Springer International Publishing, Cham, 2020.

[2] Michael Galarnyk.Considerations for Deploying Machine Learning Models in Production, November 2021.

[3] Janis Klaise, Arnaud Van Looveren, Clive Cox, Giovanni Vacanti, and Alexandru Coca. Moni-
toring and explainability of models in production, July 2020. arXiv:2007.06299 [cs, stat].

[4] Radhika V. Kulkarni, Arjun Thakur, Shravani Nalbalwar, Shubham Shah, and Sankalp Chordia. Exploring Scalable and Efficient Deployment of Machine Learning Models:A Comparative Analysis of Amazon SageMaker and Heroku. In2023 International Conference on Information Technology (ICIT), pages 746–751, Amman, Jordan, August 2023. IEEE.

[5] Philipp Mayring.Qualitative content analysis - theoretical foundation, basic procedures and software solution. January 2014.

[6] Moses Openja, Forough Majidi, Foutse Khomh, Bhagya Chembakottu, and Heng Li. Studying the Practices of Deploying Machine Learning Projects on Docker. InThe International Conference on Evaluation and Assessment in Software Engineering 2022, pages 190–200, June 2022. arXiv:2206.00699 [cs].

[7] Carolyn B. Seaman.Qualitative Methods.In Forrest Shull, Janice Singer, and Dag I. K. Sjøberg, editors,Guide to Advanced Empirical Software Engineering, pages 35–62. Springer, London, 2008.

[8] Ralf Seppelt, Felix Müller, Boris Schröder, and Martin Volk. Challenges of simulating complex environmental systems at the landscape scale:A controversial dialogue between two cups of espresso.Ecological Modelling, 220(24):3481–3489, December 2009.

[9] Hannah Snyder.Literature review as a research methodology:An overview and guidelines. Journal of Business Research, 104:333–339, November 2019.

[10] Richard J. Torraco. Writing Integrative Literature Reviews: Guidelines and Examples - Richard J. Torraco, 2005.

[11] H. Borges, A. Hora, and M. T. Valente, “Understanding the factors that impact the popularity of GitHub repositories,” in Int’l Conf. Software Maintenance and Evolution. IEEE, 2016, pp. 334–344.
