# !/usr/bin/env bash
set -euo pipefail


# echo "🗑️  Deleting existing Minikube cluster..."
minikube delete || echo "  > No existing cluster to delete."



# echo "🚀 Starting a fresh Minikube cluster..."
minikube start --memory=4096 --cpus=4


# # # # Adjust this path if your persistent_storage lives elsewhere
# HOST_PERSISTENT_DIR="$HOME/Documents/SPE Major/NST_app/persistent_storage"

# echo "🔐 Mounting host persistent_storage into Minikube..."
# # Run in background so the rest of the script can continue
# minikube mount "$HOST_PERSISTENT_DIR":/persistent_storage \
#   --uid=$(id -u) --gid=$(id -g) > /tmp/minikube-mount.log 2>&1 &



echo "🔐 Mounting host persistent_storage into Minikube..." && minikube mount "/home/deepanshu/Documents/SPE Major/NST_app/persistent_storage":/persistent_storage --uid=$(id -u) --gid=$(id -g) > /tmp/minikube-mount.log 2>&1 &



eval $(minikube docker-env)
kubectl create namespace elk
# 1. Elasticsearch
kubectl apply -f kubernetes/elk/elasticsearch-deployment.yaml
kubectl apply -f kubernetes/elk/elasticsearch-service.yaml

# 2. Logstash (ConfigMap first, then deployment)
kubectl apply -f kubernetes/elk/logstash-configmap.yaml
kubectl apply -f kubernetes/elk/logstash-deployment.yaml

# 3. Kibana
kubectl apply -f kubernetes/elk/kibana-deployment.yaml
kubectl apply -f kubernetes/elk/kibana-service.yaml

# 4. Filebeat (RBAC first, then config, then DaemonSet)
kubectl apply -f kubernetes/elk/filebeat-rbac.yaml
kubectl apply -f kubernetes/elk/filebeat-config.yaml
kubectl apply -f kubernetes/elk/filebeat-daemonset.yaml

kubectl get pods -n elk -w
kubectl describe pod pod-name -n elk
kubectl delete pods --all -n elk

kubectl delete all --all -n elk




eval $(minikube docker-env)
docker-compose up --build

echo "⏱️  Waiting 3 seconds for mount to be established..."
sleep 3

echo "📦 Applying PersistentVolume and PersistentVolumeClaim..."
kubectl apply -f kubernetes/persistent-volumes/pv.yaml
kubectl apply -f kubernetes/persistent-volumes/pvc.yaml

echo "📦 Deploying microservices..."
kubectl apply -R -f kubernetes/deployments
kubectl apply -R -f kubernetes/services

minikube addons enable ingress  

echo "📎 Applying Ingress configuration..."
kubectl apply -f kubernetes/ingress/ingress.yaml

echo
echo "🏷  Current cluster status:"
kubectl get all --all-namespaces

echo
echo "🌐 Minikube IP (use this in your browser):"
MINIKUBE_IP=$(minikube ip)
echo "    e.g. http://$MINIKUBE_IP.nip.io"

echo
echo "🎉 Done! If things didn’t come up, inspect logs with 'kubectl describe' or 'kubectl logs'."


minikube kubectl -- config view --raw > minikube-kubeconfig.yaml