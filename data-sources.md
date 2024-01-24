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

* Number of supported input model type: it can deploy any common model built with popular ML training frameworks. e.g., onnx, tensorflow, keras, etc. [[2]](#References)

  * AWS SageMaker
  * Azure ML (https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-catalog?view=azureml-api-2#collections):
    * **Open source models curated by Azure AI**, 'Curated by Azure AI' and collections from partners such as **Meta, NVIDIA, Mistral AI** are all curated collections on the Catalog.
    * **Azure OpenAI models, exclusively available on Azure**
    * **Transformers models from the HuggingFace hub**
  * BentoML: As per their [documentation](), BentoML supports the following ML frameworks:
    * CatBoost
    * Diffusers
    * fast.ai
    * Keras
    * LightGBM
    * MLflow
    * ONNX
    * PyTorch
    * PyTorch Lightning
    * Scikit-Learn
    * TensorFlow
    * Transformers
    * XGBoost
    * Detectron2
    * EasyOCR.
      Apart from these, BentoML also supports [custom models](https://docs.bentoml.org/en/latest/frameworks/index.html#custom-models).
  * Kubeflow:
    * **PyTorch**
    * **TensorFlow**
    * **Hugging Face**
    * **Jupyter**
    * **scikit-learn**
    * **HOROVOD**
    * **dmlcXGBoost**
  * MLflow: It supports many ML frameworks as [built-in flavours](https://mlflow.org/docs/latest/models.html#built-in-model-flavors), see here for list. Apart from these, it also supports [community contributed flavours](https://mlflow.org/docs/latest/community-model-flavors.html).
  * RayServe
  * Seldon Core: it supports the most common ML frameworks, see [here](https://docs.seldon.io/projects/seldon-core/en/v2/contents/models/inference-artifacts/index.html) for details. It also supports custom models e.g., pickled models.
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Model Metrics Monitoring: As it is not possible to check the output labels for live
  data, the statistics of input data and output predictions are monitored as a proxy for model performance. [3]

  * AWS SageMaker
  * Azure ML: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2)
  * BentoML:
    * Yes, it does provide an unified interface for logging model metrics and predictions to various platforms. See [here](https://docs.bentoml.org/en/latest/guides/monitoring.html) for details.
  * Kubeflow: Yes (via Alibi-detect from Seldon Core: https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md)
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* **Anomaly detection**: Also known as **outlier detection**, it is the process of identifying data points that deviate to a large extent from the overall pattern of the data. This is crucial in machine learning (ML) deployment, as it helps identify anomalies in the input data that could negatively impact model performance. [[3]](#References) Additionally, **data drift detection** is part of this criterion, data drift occurs when changes are observed in joint distribution $p(X, y)$, where $X$ is the model input and $y$ is the model output. Detection of data drift is a growing concern for teams that maintain ML models in production [[8]](#References).

  * AWS SageMaker
  * Azure ML: Yes (https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/introducing-azure-anomaly-detector-api/ba-p/490162)
    * Outlier Detection (Change Points detection): Yes (https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview)
    * Data Drift Detection: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?view=azureml-api-1&tabs=python)
  * BentoML:
    * Yes, but there is a caveat. BentoML provides an unified interface for logging model metrics but their [blog article](https://bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection) mentions that it relies on other open-source libraries for the feature of data drift detection. The documentation didn't mention any specific library for outlier detection.
  * Kubeflow: Yes (via Alibi-detect from Seldon Core: https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md)
  * MLflow
  * RayServe
  * Seldon Core: Yes
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Model explainability (might not be relevant for deployment tools): Explanation algorithms should be available in order to provide insights into the decision process of the model. Explainability in the outputs of a deployed model help in building trust in the ML system [[3]](#References).

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
* Continuous integration and continuous delivery: CI/CD is a powerful tool that can help ML teams develop and deploy models faster and more efficiently [[8]](#References).

  * AWS SageMaker
  * Azure ML: Yes (https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-event-grid?view=azureml-api-2)
  * BentoML: Yes, it is possible via [GitHub Actions](https://docs.bentoml.com/en/latest/guides/github-actions.html). A guide to setting up an automated CI/CD pipeline for [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) deployment can be found [here](https://docs.bentoml.org/en/latest/bentocloud/best-practices/bento-building-and-deployment.html).
  * Kubeflow:
  * MLflow
  * RayServe
  * Seldon Core
  * TensorFlow Serving
  * TorchServe
  * Vertex AI
* Deployment Focus: Which kind of ML models does the option support to deploy, e.g. Only Machine Learning models or Web applications possible.[[4]](#References)

  * AWS SageMaker
  * Azure ML: any kind of ML models
  * BentoML: Even though BentoML is designed with ML models in mind, it doesn't enforce the inclusion of a model. You can simply package your web application's code and dependencies in a Bento.
  * Kubeflow:
  * MLflow
  * RayServe: any kind of ML models ([Ray Serve: Scalable and Programmable Serving — Ray 2.9.1](https://docs.ray.io/en/latest/serve/index.html#ray-serve-scalable-and-programmable-serving))
  * Seldon Core
  * TensorFlow Serving: only TensorFlow models
  * TorchServe
  * Vertex AI: only ML models built with TensorFlow, XGBoost, Scikit-learn, or custom containers, as online or batch prediction services.
* Cost plan: How much does it cost e.g. monthly?[[4]](#References) We can find this data from their pricing page of the respective websites. This can also include feature like cost monitoring.

  * AWS SageMaker: pay-as you-go
  * Azure ML: consumption-based
  * BentoML: Open-source. (ask about including paid plans of subsidiaries like BentoCloud to Markus)
    It does offers fully managed platforms on a pay-as-you-go basis. It has a enterprise plan that offers additional features like deploying on VPCs, dedicated support, etc. See [details](https://www.bentoml.com/cloud#pricing).
  * Kubeflow: open-source
  * MLflow: open-source
  * RayServe: open-source
  * Seldon Core: open-source
  * TensorFlow Serving: open-source
  * TorchServe: open-source
  * Vertex AI: pay-as you-go
* Compatibility with Docker/Docker support: Docker containers encapsulate all the dependencies, libraries, and configurations needed for an application. It also allows for the creation of reproducible environments. This means that the same Docker container can be run on different machines or cloud services, ensuring that the ML model’s behavior remains consistent across various deployment scenarios. In many cases, platforms can offer pre-built docker images for common use-cases. [[6]](#References)
  (it might not be a good criterion as all the options support Docker)

  * AWS SageMaker: Yes ([Use Docker containers to build models - Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html))
  * Azure ML: Yes ([microsoft-azureml - Official Image | Docker Hub](https://hub.docker.com/_/microsoft-azureml))
  * BentoML
  * Kubeflow: Yes ([Container Images | Kubeflow](https://www.kubeflow.org/docs/components/notebooks/container-images/))
  * MLflow: Yes ([Official MLflow Docker Image — MLflow 2.9.2 documentation](https://mlflow.org/docs/latest/docker.html))
  * RayServe: Yes ([Trying to deploy ray with docker - Ray Libraries (Data, Train, Tune, Serve) / Ray Serve - Ray](https://discuss.ray.io/t/trying-to-deploy-ray-with-docker/766))
  * Seldon Core
  * TensorFlow Serving: Yes ([TensorFlow Serving with Docker  |  TFX](https://www.tensorflow.org/tfx/serving/docker))
  * TorchServe
  * Vertex AI: Yes ([Custom containers overview  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/training/containers-overview))
* Offline batch processing/Request batching: it refers to performing predictions on a batch of data in a scheduled, non-interactive, and often offline manner. Some models are not deployed for real-time applications, they can utilize the parallelizing capabilities of hardware accelerators to wait for a batch of requests to accumulate and then complete them together. [[2]](#References)

  * AWS SageMaker
  * Azure ML: Yes ([Deploy machine learning models in production environments - Cloud Adoption Framework | Microsoft Learn](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/innovate/best-practices/ml-deployment-inference#architectural-considerations))
  * BentoML: Yes, it supports adaptive batching. See [here](https://docs.bentoml.com/en/latest/guides/batching.html) for details.
  * Kubeflow: Yes (via Seldon Core: [Batch processing with Kubeflow Pipelines — seldon-core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_pipelines_batch.html))
  * MLflow
  * RayServe: Yes ([Dynamic Request Batching — Ray 2.9.1](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html))
  * Seldon Core: Yes ([Batch processing with Kubeflow Pipelines — seldon-core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_pipelines_batch.html))
  * TensorFlow Serving: Yes ([serving/tensorflow_serving/batching/README.md at master · tensorflow/serving · GitHub](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md#batch-scheduling-parameters-and-tuning))
  * TorchServe
  * Vertex AI ([Get batch predictions and explanations  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions))

## References

[1] Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. Rapid Reviews in Software Engineering. In
Michael Felderer and Guilherme Horta Travassos, editors,Contemporary Empirical Methods in
Software Engineering, pages 357–384. Springer International Publishing, Cham, 2020.

[2] Michael Galarnyk.Considerations for Deploying Machine Learning Models in Production,
November 2021.

[3] Janis Klaise, Arnaud Van Looveren, Clive Cox, Giovanni Vacanti, and Alexandru Coca. Moni-
toring and explainability of models in production, July 2020. arXiv:2007.06299 [cs, stat].

[4] Radhika V. Kulkarni, Arjun Thakur, Shravani Nalbalwar, Shubham Shah, and Sankalp Chordia.
Exploring Scalable and Efficient Deployment of Machine Learning Models:A Comparative
Analysis of Amazon SageMaker and Heroku. In2023 International Conference on Information
Technology (ICIT), pages 746–751, Amman, Jordan, August 2023. IEEE.

[5] Philipp Mayring.Qualitative content analysis - theoretical foundation, basic procedures and
software solution. January 2014.

[6] Moses Openja, Forough Majidi, Foutse Khomh, Bhagya Chembakottu, and Heng Li. Studying
the Practices of Deploying Machine Learning Projects on Docker. InThe International Confer-
ence on Evaluation and Assessment in Software Engineering 2022, pages 190–200, June 2022.
arXiv:2206.00699 [cs].

[7] Carolyn B. Seaman.Qualitative Methods.In Forrest Shull, Janice Singer, and Dag I. K.
Sjøberg, editors,Guide to Advanced Empirical Software Engineering, pages 35–62. Springer,
London, 2008.

[8] Ralf Seppelt, Felix Müller, Boris Schröder, and Martin Volk. Challenges of simulating complex
environmental systems at the landscape scale:A controversial dialogue between two cups of
espresso.Ecological Modelling, 220(24):3481–3489, December 2009.

[9] Hannah Snyder.Literature review as a research methodology:An overview and guidelines.
Journal of Business Research, 104:333–339, November 2019.

[10] Richard J. Torraco. Writing Integrative Literature Reviews: Guidelines and Examples - Richard
J. Torraco, 2005.
