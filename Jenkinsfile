// ============================================================
// Jenkinsfile - Pipeline de validacion del proyecto
// Este archivo usa "bat" (Windows). Si tu Jenkins esta en Linux/Mac/Docker,
// cambia cada  bat '...'  por  sh '...'  y usa  .venv/bin/python  en vez de  .venv\Scripts\python
// ============================================================
pipeline {
    agent any

    // TRIGGER AUTOMATICO: Jenkins revisa el repositorio cada ~5 minutos y
    // ejecuta el pipeline si hay commits nuevos (por ejemplo, tras un merge).
    triggers {
        pollSCM('H/5 * * * *')
    }

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                checkout scm
            }
        }

        stage('Preparar entorno') {
            steps {
                bat 'python -m venv .venv'
                bat '.venv\\Scripts\\python -m pip install --upgrade pip'
                bat '.venv\\Scripts\\python -m pip install -r requirements.txt'
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                // Si alguien cambia o borra un archivo de la estructura, el pipeline falla aqui
                bat '.venv\\Scripts\\python -m pytest mlops_pipeline/test_estructura.py -q'
            }
        }

        stage('Pruebas unitarias y cobertura') {
            steps {
                bat '.venv\\Scripts\\python -m pytest mlops_pipeline --cov --cov-report=xml'
            }
        }
    }

    // NOTIFICACION al terminar (requiere configurar el correo SMTP en Jenkins:
    // Administrar Jenkins > System > E-mail Notification)
    post {
        success {
            mail to: 'tu_correo@ejemplo.com',
                 subject: "Pipeline OK: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "El pipeline termino correctamente.\n${env.BUILD_URL}"
        }
        failure {
            mail to: 'tu_correo@ejemplo.com',
                 subject: "Pipeline FALLO: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "El pipeline fallo. Revisa el detalle aqui:\n${env.BUILD_URL}"
        }
    }
}
