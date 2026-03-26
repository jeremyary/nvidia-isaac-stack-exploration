{{/*
Common labels
*/}}
{{- define "isaac-mlops.labels" -}}
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/part-of: isaac-mlops-poc
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "isaac-mlops.namespace" -}}
{{ .Values.namespace }}
{{- end }}

{{/*
GPU tolerations
*/}}
{{- define "isaac-mlops.gpuTolerations" -}}
{{- range .Values.gpu.tolerations }}
- key: {{ .key }}
  operator: {{ .operator }}
  effect: {{ .effect }}
{{- end }}
{{- end }}

{{/*
GPU node selector
*/}}
{{- define "isaac-mlops.gpuNodeSelector" -}}
nvidia.com/gpu.product: {{ .Values.gpu.product }}
{{- end }}
