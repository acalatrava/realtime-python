

# Asistente de Voz para la API en Tiempo Real de OpenAI

Este proyecto en Python implementa un asistente de voz utilizando la nueva API en Tiempo Real de OpenAI. Cuenta con un sistema de Detección de Actividad de Voz (VAD) del lado del cliente para optimizar el uso de tokens y reducir costos.

## Características

- Utiliza la API en Tiempo Real de OpenAI para conversaciones en tiempo real
- Implementa Detección de Actividad de Voz (VAD) del lado del cliente
- Admite tanto modalidades de texto como de audio
- Proporciona entrada y salida de audio en tiempo real
- Calcula y muestra el uso de tokens y los costos asociados
- Permite detener al asistente cuando vuelvas a hablar

## Video de ejemplo

https://github.com/user-attachments/assets/c40001bc-198b-4f5b-b98a-49b8984601ed

## ¿Por qué VAD del lado del cliente?

Esta implementación utiliza la Detección de Actividad de Voz (VAD) del lado del cliente, lo cual ofrece varias ventajas:

1. **Eficiencia de Costos**: Al enviar audio a OpenAI únicamente cuando se detecta voz, reduces significativamente la cantidad de tokens procesados, disminuyendo los costos de uso de la API.

2. **Latencia Reducida**: El VAD del lado del cliente permite tiempos de respuesta más rápidos, ya que no depende del procesamiento del servidor para determinar cuándo termina el habla.

3. **Optimización de Ancho de Banda**: Solo se transmiten datos de audio relevantes, reduciendo el consumo de ancho de banda.

## Para comenzar

1. Clona este repositorio
2. Instala las dependencias requeridas:
   ```
   pip install -r requirements.txt
   ```
3. Configura tu clave de API de OpenAI en el archivo `.env`
4. Ejecuta la aplicación:
   ```
   python start.py
   ```

## Configuración

Puedes personalizar varios ajustes en el archivo `start.py`, incluyendo:

- Umbral de silencio para el VAD
- Duración mínima de silencio

También puedes modificar el archivo `prompt.txt` para cambiar el prompt del asistente.

## Uso

Después de iniciar la aplicación, habla en tu micrófono. El sistema detectará tu voz, procesará tu habla y proporcionará respuestas tanto en texto como en audio del asistente de IA.

## Nota

Este proyecto está diseñado con fines educativos y experimentales. Asegúrate de mantener tu clave de API privada y no exponerla al público.

## Licencia

[MIT License](LICENSE)
