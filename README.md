# kubeflow-GNN

- Project goal - workings of diffrent components of kubeflow (ML toolkit for kubernetes) by training and serving GNN(graph neural network) model for link-prediction on Dataset [ogbl-citation2](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2)

---

## kubeflow

- **what is it?**

  - "The Kubeflow project is dedicated to making deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable. Our goal is not to recreate other services, but to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures. Anywhere you are running Kubernetes, you should be able to run Kubeflow." - [kubeflow](https://www.kubeflow.org/docs/started/introduction/)

- **kubeflow in ML workflows stages**
  - data preparation
  - model training
  - prediction serving
  - service management
- \*\*

## steps

- **kubeflow setup**
  - deploy kubeflow in any cloud (GCP/AWS/AZURE) or setup in vanila kube cluster.
- **setup docker credentials**
  - purpose - Kaniko is used by fairing to build the model every time the notebook is run and deploy a fresh model. The newly built image is pushed into the DOCKER_REGISTRY and pulled from there by subsequent resources.
- **launch a jupyter notebook**
  - will contain code for train and serve model, as well as pushing the docker image in registry / minio bucket.

## glossary

- **kaniko tool**
  - kaniko is a tool to build container images from a Dockerfile, inside a container or Kubernetes cluster. kaniko doesn't depend on a Docker daemon and executes each command within a Dockerfile completely in userspace. This enables building container images in environments that can't easily or securely run a Docker daemon, such as a standard Kubernetes cluster.
- **kubeflow fairing**
  - Kubeflow Fairing streamlines the process of building, training, and deploying machine learning (ML) training jobs in a hybrid cloud environment. By using Kubeflow Fairing and adding a few lines of code, you can run your ML training job locally or in the cloud, directly from Python code or a Jupyter notebook. After your training job is complete, you can use Kubeflow Fairing to deploy your trained model as a prediction endpoint.

steps

- create the kube cluster
- kube cluster is working

- setup of the kubeflow in kubecluster
- i will test using mnist code
- in mnist i have to build an image and then push it to the gcr
- gcr will have that and then need to deploy job in kube cluster
