# Minuta de Reunión Kick-off - Proyecto Predicción Enfermedades Cardíacas

**Fecha:** Semana 3 del Proyecto
**Asistentes:** Jhiens, Joel, Luiggi, Emhir, Hermoza

## 1. Objetivos de la Reunión
* Definir roles y responsabilidades del equipo.
* Establecer el flujo de trabajo y estrategia de control de versiones (Git).
* Alinear la visión del alcance del Sprint 1.

## 2. Definición de Roles

Basado en la planificación del proyecto, se asignan los siguientes roles:

| Rol | Responsabilidades Principales | Asignado a |
| :--- | :--- | :--- |
| **Coordinador** | Gestión del cronograma, entregas y comunicación general. | *Por definir* |
| **Líder Técnico** | Arquitectura del sistema, estándares de código, configuración del entorno. | *Por definir* |
| **Ingeniero de ML** | Diseño e implementación de algoritmos (XGBoost), optimización matemática. | *Por definir* |
| **Ingeniero de Datos** | Ingesta, limpieza, transformación de datos (.XPT a CSV/Parquet) y EDA inicial. | *Por definir* |
| **Analista de IA** | Estado del arte, interpretabilidad (SHAP/LIME), análisis ético y visualización profunda. | *Por definir* |
| **Desarrollador UI** | Implementación de la interfaz de usuario (Streamlit). | *Por definir* |
| **Documentador** | Redacción de informes, manuales y documentación técnica. | *Por definir* |

> *Nota: Un integrante puede cubrir más de un rol.*

## 3. Estrategia de Trabajo (Git Flow)

Para evitar conflictos y asegurar la calidad del código, se adoptará la siguiente estrategia de ramas:

*   **Ramas Principales:**
    *   `main`: Código en producción, estable y probado. **Prohibido hacer commit directo.**
    *   `develop`: Rama de integración principal. Aquí se fusionan las funcionalidades terminadas.

*   **Ramas de Trabajo:**
    *   `feature/<nombre-tarea>`: Para nuevas funcionalidades (ej: `feature/setup-entorno`, `feature/implementacion-xgboost`).
    *   `fix/<bug>`: Para corrección de errores.
    *   `docs/<nombre-doc>`: Para documentación.

*   **Reglas de Juego:**
    1.  Cada integrante crea una rama desde `develop` (o `main` al inicio).
    2.  Se trabaja en archivos separados según el rol (Ing. Datos en `data/`, Ing. ML en `src/`, etc.).
    3.  Al terminar, se abre un **Pull Request (PR)** hacia `develop`.
    4.  **Code Review:** Otro miembro del equipo debe revisar y aprobar el PR antes de fusionar.
    5.  Se deben pasar los checks automáticos (pre-commit hooks) antes de subir cambios.

## 4. Acuerdos del Sprint 1
*   **Prioridad:** Entender la teoría de XGBoost antes de codificar.
*   **Entregable Clave:** Documentación de la función objetivo y regularización.
*   **Configuración:** Todos deben instalar el entorno Conda `xgb_env` y configurar `pre-commit`.

## 5. Próximos Pasos
*   Configurar repositorios locales.
*   Iniciar investigación bibliográfica para Issues 4, 5 y 6.
