# Tsai-Python
    Autores: @dbuitrago24 @kevin-ortega2724 @DnelZpt
# Calibración de Cámara 

## Descripción
Este proyecto aborda la calibración de una cámara utilizando el método de Tsai, seguido de la corrección de distorsión radial con OpenCV, y finaliza con el desarrollo de una aplicación de demostración para validar la calibración.

## Proceso de Calibración

### Calibración con el Método de Tsai

1. **Preparación del Aparato de Tsai**:
   Configuración del patrón de calibración.

2. **Captura de Imágenes**:
   Las imágenes  que se muestran a continuación, presentan los ejemplos en la captura de las imágenes antes de la calibración de la cámara.
<p align="center">
  <img src="shoots/1.jpg" alt="Imagen1" width="30%" style="border: 2px solid black;"/>
  <img src="shoots/2.jpg" alt="Imagen2" width="30%" style="border: 2px solid black;"/>
  <img src="shoots/3.jpg" alt="Imagen3" width="30%" style="border: 2px solid black;"/>
</p>



   Imágenes capturadas desde diferentes ángulos.


## Proceso de Etiquetado de Imágenes con Patrón de Ajedrez

El proceso de etiquetado de las imágenes que contienen el patrón de ajedrez es un paso crucial para la calibración precisa de la cámara. A continuación se detallan los pasos seguidos:

1. **Selección de Imágenes**:
   - Estas imágenes contienen el patrón de ajedrez visible desde diferentes ángulos como se evidencia previamente.

2. **Identificación de Esquinas**:

   - Para el procesamiento de imágenes en este proyecto, la necesidad de software avanzado como OpenCV es fundamental para identificar automáticamente las esquinas del patrón de ajedrez en cada imagen.

Se observan algunas imágenes a continuación:
<p align="center">
  <img src="data/img_01.png" alt="Imagen1" width="30%" style="border: 2px solid black;"/>
  <img src="data/img_02.png" alt="Imagen2" width="30%" style="border: 2px solid black;"/>
  <img src="data/img_03.png" alt="Imagen3" width="30%" style="border: 2px solid black;"/>
</p>
Imagenes etiquetadas

### Funciones Principales del Proyecto

1. **`load_image(file_path)`**:
   - Carga una imagen en escala de grises desde la ruta de archivo especificada.
   - Utiliza `cv2.imread` con el flag `cv2.IMREAD_GRAYSCALE`.

2. **`get_corner_points(image)`**:
   - Obtiene los puntos de esquina de una imagen utilizando `cv2.goodFeaturesToTrack`.
   - Retorna los puntos de esquina como un array de NumPy.

3. **`show_corners(image, corners)`**:
   - Visualiza los puntos de esquina en una imagen.
   - Marca cada punto de esquina con `cv2.circle` y `cv2.putText`.

4. **`corners_report(corners)`**:
   - Imprime un reporte de los puntos de esquina encontrados.
   - Muestra el número de esquinas y sus coordenadas x, y.


3. **Verificación Manual**:
   - La verificación manual asegura que todas las esquinas hayan sido identificadas correctamente.
   - Además corrige cualquier error en la detección automática para garantizar la precisión en el proceso de calibración.

4. **Almacenamiento de Coordenadas**:
   - Es importante mantener una estructura de datos consistente para facilitar el procesamiento posterior.

5. **Repetición para Todas las Imágenes**:
   - Este proceso se repite para todas las imágenes seleccionadas en la carpeta `shoots`.

Este proceso asegura que se tenga la información necesaria sobre la posición de las esquinas del patrón en cada imagen. 





### Desarrollo de una Aplicación de Demostración

1. **Visualización de los parámetros Extrínsecos**:
   
   
<p align="center">
  <img src="extrinsec.svg" alt="Imagen1" width="40%" style="border: 2px solid black;"/>
    <img src="extrinsec2.svg" alt="Imagen2" width="40%" style="border: 2px solid black;"/>
</p>

2. **Selección de una Aplicación**:
   Verificación de parámetros extrínsecos con una imaen del mundo real. 

<p align="center">
  <img src="test.jpg" alt="Imagen1" width="40%" style="border: 2px solid black;"/>
</p>

Este procedimiento facilita la confirmación de la precisión en la calibración realizada. Se emplea una imagen de control que no se incluyó en el conjunto de datos de calibración. De manera análoga al método previamente empleado, es imprescindible la etiqueta de los puntos característicos en la imagen para permitir una comparación efectiva entre las proyecciones calculadas y las posiciones reales de estos puntos. Esto es necesario para validar la precisión de la calibración, ya que permite identificar y cuantificar cualquier desviación o error en la proyección. Al correlacionar los puntos anotados en la imagen de prueba con sus respectivas coordenadas 3D en el espacio, se pueden calcular los errores de reproyección y evaluar la calidad del modelo de calibración. Si los errores son mínimos y dentro de los márgenes aceptables, se confirma que la calibración es precisa y fiable para aplicaciones posteriores. En caso contrario, se requerirá una revisión y ajuste de los parámetros de calibración para mejorar la precisión.



3. **Implementación del proyecto**:

    Se muestran imágenes demos.
<p align="center">
  <img src="test_lines.jpg" alt="Imagen1" width="40%" style="border: 2px solid black;"/>
</p>

