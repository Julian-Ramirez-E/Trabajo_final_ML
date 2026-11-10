// ============================================================
// Jenkinsfile - Pipeline de validacion del proyecto
// Funciona en Linux/Docker (sh) y en Windows (bat): detecta el sistema solo.
// ============================================================

// ---- Funciones auxiliares (fuera del pipeline) ----
// Ejecuta un comando con sh (Linux) o bat (Windows)
def ejecutar(String comando) {
    if (isUnix()) {
        sh comando
    } else {
        bat comando
    }
}

// Ruta del Python del entorno virtual segun el sistema operativo
def python() {
    return isUnix() ? '.venv/bin/python' : '.venv\\Scripts\\python'
}

// Notificacion final: siempre queda escrita en el log; el correo se intenta
// y, si el SMTP no esta configurado en Jenkins, el pipeline NO falla por eso.
def notificar(String estado, String detalle) {
    echo "NOTIFICACION [${estado}] ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${detalle} ${env.BUILD_URL}"
    try {
        mail to: 'tu_correo@ejemplo.com',
             subject: "Pipeline ${estado}: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
             body: "${detalle}\n${env.BUILD_URL}"
    } catch (err) {
        echo "No se pudo enviar el correo (SMTP sin configurar): ${err.message}"
    }
}

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
                script {
                    ejecutar(isUnix() ? 'python3 -m venv .venv' : 'python -m venv .venv')
                    ejecutar("${python()} -m pip install --upgrade pip")
                    ejecutar("${python()} -m pip install -r requirements.txt")
                }
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                // Si alguien cambia o borra un archivo de la estructura, el pipeline falla aqui
                script {
                    ejecutar("${python()} -m pytest mlops_pipeline/test_estructura.py -q")
                }
            }
        }

        stage('Pruebas unitarias y cobertura') {
            steps {
                script {
                    ejecutar("${python()} -m pytest mlops_pipeline --cov --cov-report=xml")
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true
        }
        success {
            script { notificar('OK', 'El pipeline termino correctamente.') }
        }
        failure {
            script { notificar('FALLO', 'El pipeline fallo. Revisa el detalle en:') }
        }
    }
}
