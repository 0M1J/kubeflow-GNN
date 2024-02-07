# Overview
This project aims to train a neural network for predicting missing citations using a distributed system. The system utilizes Kubeflow, an open-source project for deploying and managing machine learning workflows on Kubernetes. With Kubeflow, the project employs distributed training with PyTorch on a Kubernetes cluster to handle large-scale machine learning tasks efficiently. The system's architecture and functionalities are detailed below.

## System Architecture

### Components Used:
- **Dataset**: Utilizes the ogbl-citations2 dataset, a directed graph representing citation networks between papers.
- **GNN Model**: Implements GraphSage GNN for generating node embeddings and a Link Predictor for citation prediction.
- **PyTorch Distributed Training**: Utilizes DistributedDataParallel (DDP) for data parallelism across multiple machines.
- **Kubeflow**: Utilizes various Kubeflow resources such as MinIO for object storage, Kubeflow Pipeline Deployer for deploying ML pipelines, and MySQL for storing experiment metadata.
- **Kubernetes**: Utilizes Google Kubernetes Engine (GKE) API for Kubernetes cluster provisioning and management. Key Kubernetes resources include Deployments, Jobs, Services, Persistent Volumes, and Persistent Volume Claims.
- **TensorBoard**: Provides visualization and tooling for tracking and visualizing metrics during machine learning experiments.

## Demo Scenario
### Features Demonstrated:
- **Scalability & Performance**: Experimented with different numbers of worker pods and epochs to measure training time and accuracy.
- **Resource Sharing**: Monitored CPU, memory, and file system usage across pods. Shared file system using NFS for TensorBoard visualization.
- **Fault Tolerance**: Demonstrated fault tolerance by simulating pod failure during training, showcasing Kubernetes' auto-restart capability.

## Performance Results
### Training Performance:
| No. of Workers | Epoch | Time Taken | Accuracy |
|----------------|-------|------------|----------|
| 2              | 10    | 12.18 min  | 73.82%   |
| 4              | 10    | 12.04 min  | 73.74%   |
| 6              | 10    | 11.17 min  | 74.28%   |
| 2              | 30    | 35.45 min  | 83.20%   |
| 4              | 30    | 33.93 min  | 85.04%   |
| 6              | 30    | 30.76 min  | 85.11%   |

## Challenges Faced and Solutions
- **Resource Management**: Managing resource allocation and utilization across multiple pods and nodes was a significant challenge. We addressed this by carefully monitoring resource usage, optimizing resource requests and limits, and scaling resources as needed.
- **Networking and Communication**: Ensuring efficient communication between distributed components, especially in a cloud environment, posed challenges related to network latency and reliability. We optimized network configurations and implemented retry mechanisms to mitigate these issues.
- **Fault Tolerance**: Ensuring system resilience in the face of failures required careful planning and implementation of fault-tolerant strategies. We leveraged Kubernetes' built-in features for automatic pod restarts and implemented application-level fault tolerance mechanisms to handle failures gracefully.
- **Shared Data Storage**: Initially, we faced challenges in setting up shared data storage for TensorBoard visualization. We opted to use Nginx as a reverse proxy to serve TensorBoard data, ensuring seamless access to visualization logs across pods.
- **Image Building and Deployment**: Building Docker images and storing them in the registry using Google Cloud Build was a key requirement. We configured Google Cloud Build to automatically build Docker images from our source code repository and store them in the Google Container Registry, streamlining the deployment process.
- **GNN Training Pipelines**: Implementing GNN training pipelines required understanding the complex architecture of the GNN model and integrating it into the distributed training framework. We carefully designed and tested the training pipelines to ensure efficient distributed training across multiple nodes.


## Usage
1. **Dataset**: Download and preprocess the ogbl-citations2 dataset.
2. **Model Training**: Train the GNN model using PyTorch distributed training on a Kubeflow-enabled Kubernetes cluster.
3. **TensorBoard Visualization**: Visualize training metrics using TensorBoard.
4. **Demo Scenarios**: Experiment with different configurations for scalability, performance, resource sharing, and fault tolerance.

## Conclusion
This project demonstrates the capabilities of Kubeflow and Kubernetes in deploying distributed machine learning workflows. It showcases scalability, performance, resource sharing, and fault tolerance, essential for tackling large-scale machine learning tasks.

## References
- [Kubeflow: Official Website](https://www.kubeflow.org/)
- [PyTorch Distributed: PyTorch Documentation](https://pytorch.org/docs/stable/distributed.html)
- [Kubernetes: Official Website](https://kubernetes.io/)
