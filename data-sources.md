## Deployment Options

* [AWS Sagemaker](https://aws.amazon.com/sagemaker/)
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

* Supported ML frameworks: it can deploy any common model built with popular ML training frameworks. e.g., onnx, tensorflow, keras, etc. [[2]](#References)

  * AWS SageMaker: The full list of frameworks that are supported (via SageMaker Python SDK) can be found [here](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html).
  * Azure ML: As per their [documentation](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-catalog?view=azureml-api-2#collections), Azure ML supports the following ML frameworks:

    * Open source models curated by Azure AI, 'Curated by Azure AI' and collections from partners such as Meta, NVIDIA, Mistral AI are all curated collections on the Catalog.
    * Azure OpenAI models, exclusively available on Azure
    * Transformers models from the HuggingFace hub
  * BentoML: As per their [documentation](https://docs.bentoml.org/en/latest/frameworks/index.html), BentoML supports the following ML frameworks:

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
  * Kubeflow: As per [documentation](https://www.kubeflow.org/docs/external-add-ons/serving/overview/)

    * PyTorch
    * TensorFlow
    * Hugging Face
    * Jupyter
    * scikit-learn
    * HOROVOD
    * dmlcXGBoost
  * MLflow: It supports many ML frameworks, see [built-in flavours](https://mlflow.org/docs/latest/models.html#built-in-model-flavors) for list. Apart from these, it also supports [community contributed flavours](https://mlflow.org/docs/latest/community-model-flavors.html).
  * RayServe: It supports many ML frameworks, see [documentation](https://docs.ray.io/en/latest/serve/tutorials/serve-ml-models.html#serve-ml-models-tutorial) for list, i.e.:

    * Tensorflow
    * PyTorch
    * Scikit-Learn
    * others
  * Seldon Core: it supports the most common ML frameworks, see [here](https://docs.seldon.io/projects/seldon-core/en/v2/contents/models/inference-artifacts/index.html) for details. It also supports custom models e.g., pickled models.
  * TensorFlow Serving: it supports only TensorFlow models. see [here](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
  * TorchServe: it supports only Pytorch models. See [website](https://pytorch.org/serve/)
  * Vertex AI: see list of supported frameworks [here](https://cloud.google.com/vertex-ai/docs/supported-frameworks-list?hl=de).
* Model Metrics Monitoring: As it is not possible to check the output labels for live data, the statistics of input data and output predictions are monitored as a proxy for model performance. [[3]](#References)

  * AWS SageMaker: Yes, model monitoring is a feature in AWS Sagemaker, see details [here](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html).
  * Azure ML: Yes, check [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2).
  * BentoML: Yes, it does provide an unified interface for logging model metrics and predictions to various platforms. See [here](https://docs.bentoml.org/en/latest/guides/monitoring.html) for details.
  * Kubeflow: Yes, via [Alibi-detect from Seldon Core](https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md), and [DKube](https://www.youtube.com/watch?v=xu-V13XbYCQ).
  * MLflow: Yes. It provides an API and a UI for conducting model monitoring. The tracking tool is organized around the concept of runs which can be organized into experiments. The tracking APIs provide a set of functions to track your runs and this tracked data can be later visualized in the tracking UI. See details [here](https://mlflow.org/docs/latest/tracking.html).
  * RayServe: Yes, see [here](https://docs.ray.io/en/latest/ray-observability/index.html).
  * Seldon Core: Yes, via [Alibi](https://www.seldon.io/solutions/open-source-projects/alibi-detect), see [paper](https://www.seldon.io/research/monitoring-and-explainability-of-models-in-production).
  * TensorFlow Serving: Yes, via Prometheus or [Grafana](https://grafana.com/docs/grafana-cloud/monitor-infrastructure/integrations/integration-reference/integration-tensorflow/). See unofficial [guide](https://itnext.io/monitor-deployed-tensorflow-models-with-prometheus-and-grafana-28d6135a2666).
  * TorchServe: Yes, via Grafana and [Datadog](https://www.datadoghq.com/blog/ai-integrations/#model-serving-and-deployment-vertex-ai-amazon-sagemaker-torchserve) as mentioned on the [website](https://github.com/pytorch/serve).
  * Vertex AI: Yes, see [here](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview).
* Anomaly detection: Also known as outlier detection, it is the process of identifying data points that deviate to a large extent from the overall pattern of the data. This is crucial in machine learning (ML) deployment, as it helps identify anomalies in the input data that could negatively impact model performance. [[3]](#References) Additionally, data drift detection is part of this criterion, data drift occurs when changes are observed in joint distribution $p(X, y)$, where $X$ is the model input and $y$ is the model output. Detection of data drift is a growing concern for teams that maintain ML models in production [[8]](#References).

  * AWS SageMaker: Yes, [data quality monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html), [model quality monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html) and [bias drift monitoring in models](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-bias-drift.html) are available as features in AWS Sagemaker.
  * Azure ML: Yes, check [here](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/introducing-azure-anomaly-detector-api/ba-p/490162), Outlier Detection (Change Points detection): Yes, check [here](https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/overview), Data Drift Detection: Yes, check [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?view=azureml-api-1&tabs=python)
  * BentoML: Yes, but there is a caveat. BentoML provides an unified interface for logging model metrics but their [blog article](https://bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection) mentions that it relies on other open-source libraries for the feature of data drift detection. The documentation didn't mention any specific library for outlier detection.
  * Kubeflow: Yes, via Alibi-detect from Seldon Core, check [here](https://github.com/kubeflow/pipelines/blob/master/samples/contrib/e2e-outlier-drift-explainer/seldon/README.md)
  * MLflow: Yes, MLflow can be integrated with other libraries that can perform outlier detection.
  * RayServe: No, since it focuses on the scalable and efficient serving of machine learning models, there is no mention of such feature in the official documentation.
  * Seldon Core: Yes, via Alibi detect. Check [this link](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/outlier_detection.html) for details.
  * TensorFlow Serving: No, there is no mention of such feature in the official documentation.
  * TorchServe: No, there is no mention of such feature in the official documentation.
  * Vertex AI: Yes, see [link](https://cloud.google.com/vertex-ai/docs/featurestore/monitoring#view_feature_value_anomalies) for more details.
* Model explainability: Explanation algorithms should be available in order to provide insights into the decision process of the model. Explainability in the outputs of a deployed model help in building trust in the ML system [[3]](#References).

  * AWS SageMaker: Yes, this is implemented in AWS Sagemaker via SageMaker Clarify, see [here](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html).
  * Azure ML: Yes, model interpretability/explainability techniques are supported as documented [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-automl?view=azureml-api-1)
  * BentoML: Yes, via [Arize AI integration](https://docs.bentoml.org/en/latest/integrations/arize.html).
  * Kubeflow: Yes, this is possible via [alibi](https://www.kubeflow.org/docs/external-add-ons/serving/overview/)
  * MLflow: Yes, this is possible via [API](https://mlflow.org/docs/latest/python_api/mlflow.shap.html), it utilizes [SHAP](https://shap.readthedocs.io/en/latest/) for this function.
  * RayServe: Yes, via [Arize AI integration](https://docs.ray.io/en/latest/serve/monitoring.html#exporting-metrics-into-arize).
  * Seldon Core: Yes, it offers several algorithms for model explainability
  * TensorFlow Serving: No, there is no official documentation that mentions this feature.
  * TorchServe: Yes, offered by [captum](https://captum.ai/) library.
  * Vertex AI: Yes, this is described on the official [documentation page](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview).
* Continuous integration and continuous delivery: CI/CD is a powerful tool that can help ML teams develop and deploy models faster and more efficiently [[8]](#References).

  * AWS SageMaker: Yes, this is possible via AWS SageMaker Pipelines, see [here](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html). A [video guide](https://www.youtube.com/watch?v=bef5lHq7yLA) is also available.
  * Azure ML: Yes, check [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-event-grid?view=azureml-api-2)
  * BentoML: Yes, it is possible via [GitHub Actions](https://docs.bentoml.com/en/latest/guides/github-actions.html). A guide to setting up an automated CI/CD pipeline for [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) deployment can be found [here](https://docs.bentoml.org/en/latest/bentocloud/best-practices/bento-building-and-deployment.html).
  * Kubeflow: Yes, this is possible via [Kubeflow pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/sdk/build-pipeline/).
  * MLflow: Yes, it supports integration with CI/CD systems like Jenkins, see [here](https://www.restack.io/docs/mlflow-knowledge-mlops-mlflow-guide) for guidance.
  * RayServe: No, there is no official documentation for setting up CI/CD pipelines.
  * Seldon Core: Yes, their documentation has guides for setting up CI/CD pipelines using Jenkins Classic and Jenkins X, see [here](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/cicd-mlops.html).
  * TensorFlow Serving: Yes, check this [guide](https://blog.tensorflow.org/2022/09/automated-deployment-of-tensorflow-models-with-tensorflow-serving-and-github-actions.html) to set up automated deployment using github actions.
  * TorchServe: No, There is no official documentation for setting up CI/CD pipelines for TorchServe.
  * Vertex AI: No, there is no mention of such feature in the official documentation.
* Popularity: how popular is the framework? We quantify this metric via github stars, following the methodology of [[11](#references)]:

  * Refer to the table for star counts.
  * Numerous options in our list, including MLflow and Kubeflow, offer a broader range of functions beyond mere deployment, which may positively influence their popularity.
* Cost plan: How much does it cost e.g. monthly? [[4]](#References) We can find this data from their pricing page of the respective websites. This can also include feature like cost monitoring. TODO: why is [4] cited?

  * AWS SageMaker: Paid tool with different pricing plans, see [here](https://aws.amazon.com/sagemaker/pricing/).
  * Azure ML: Specific price not mentioned on the official website, price plan can order after consulting support.
  * BentoML: Open-source with Apache License 2.0, It does offers fully managed platforms on a pay-as-you-go basis. It has a enterprise plan that offers additional features like deploying on VPCs, dedicated support, etc. See [details](https://www.bentoml.com/cloud#pricing).
  * Kubeflow: Open-source with Apache License 2.0
  * MLflow: Open-source with Apache License 2.0
  * RayServe: Open-source with Apache License 2.0
  * Seldon Core: Open-core model. The core platform is open-source and can be used freely for non-production use cases like pre-production and testing. Production use cases require a paid license. See [source](https://www.seldon.io/strengthening-our-commitment-to-open-core) and [license](https://github.com/SeldonIO/seldon-core?tab=License-1-ov-file#readme).
  * TensorFlow Serving: Open-source with Apache License 2.0
  * TorchServe: Open-source with Apache License 2.0
  * Vertex AI: pay-as you-go
* Compatibility with Docker/Docker support: Docker containers encapsulate all the dependencies, libraries, and configurations needed for an application. It also allows for the creation of reproducible environments. This means that the same Docker container can be run on different machines or cloud services, ensuring that the ML model’s behavior remains consistent across various deployment scenarios. In many cases, platforms can offer pre-built docker images for common use-cases. [[6]](#References)

  * AWS SageMaker: Yes, it offers extensive support for docker containers, see [Use Docker containers to build models](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html) and also offers pre-built docker images for common use-cases, see [Pre-built Docker images](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html).
  * Azure ML: Yes, see [microsoft-azureml - Official Image | Docker Hub](https://hub.docker.com/_/microsoft-azureml)
  * BentoML: Yes, see [BentoML | Docker](https://docs.bentoml.org/en/latest/concepts/deploy.html#docker).
  * Kubeflow: Yes, see ([Container Images | Kubeflow](https://www.kubeflow.org/docs/components/notebooks/container-images/))
  * MLflow: Yes, it officially supports Docker and also provides an official docker image. see [Official MLflow Docker Image](https://mlflow.org/docs/latest/docker.html)
  * RayServe: Yes ([Trying to deploy ray with docker - Ray Libraries (Data, Train, Tune, Serve) / Ray Serve - Ray](https://discuss.ray.io/t/trying-to-deploy-ray-with-docker/766))
  * Seldon Core: Yes, it offers [documentation](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html) for deploying Seldon Core on Docker as well as pre-built Docker [images](https://docs.seldon.io/projects/seldon-core/en/latest/reference/images.html).
  * TensorFlow Serving: Yes, see [TensorFlow Serving with Docker  |  TFX](https://www.tensorflow.org/tfx/serving/docker)
  * TorchServe: Yes, it offer documentation for preparing docker images as well as scripts for building docker images. See [here](https://github.com/pytorch/serve/tree/master/docker) for details.
  * Vertex AI: Yes, see [Custom containers overview  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/training/containers-overview)
* Offline batch processing/Request batching: it refers to performing predictions on a batch of data in a scheduled, non-interactive, and often offline manner. Some models are not deployed for real-time applications, they can utilize the parallelizing capabilities of hardware accelerators to wait for a batch of requests to accumulate and then complete them together. [[2]](#References)

  * AWS SageMaker: Yes, see [Batch Transform - Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html).
  * Azure ML: Yes, see [Deploy machine learning models in production environments - Cloud Adoption Framework | Microsoft Learn](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/innovate/best-practices/ml-deployment-inference#architectural-considerations)
  * BentoML: Yes, it supports adaptive batching. See [here](https://docs.bentoml.com/en/latest/guides/batching.html) for details.
  * Kubeflow: Yes (via Seldon Core: [Batch processing with Kubeflow Pipelines — seldon-core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_pipelines_batch.html))
  * MLflow: No, as of now it doesn't support offline batch processing. There is an [open issue](https://github.com/mlflow/mlflow/issues/8007) for implementing opportunistic batch processing in MLflow.
  * RayServe: Yes, see [Dynamic Request Batching — Ray 2.9.1](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)
  * Seldon Core: Yes, see [Batch Processing with Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/servers/batch.html)
  * TensorFlow Serving: Yes, see [serving/tensorflow_serving/batching/README.md at master · tensorflow/serving · GitHub](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md#batch-scheduling-parameters-and-tuning)
  * TorchServe: Yes, it supports request batching as per [documentation](https://pytorch.org/serve/batch_inference_with_ts.html).
  * Vertex AI: Yes, see [Get batch predictions and explanations  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions)

## References

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
