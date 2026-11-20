"""Demo del TP2 — Evaluación y normalización de textos rioplatenses.

Corre sin API key y sin BETO: usa el normalizador por diccionario y los dos
perceptrones multicapa entrenados en el notebook (intención y sentimiento).
"""
import os
import pickle
from pathlib import Path

import gradio as gr
import pandas as pd

from normalizador import buscar_expresiones, compilar, normalizar

AQUI = Path(__file__).parent


# ---------------------------------------------------------------------
# Carga de modelos: acepta un paquete único o los dos .pkl separados
# ---------------------------------------------------------------------
def cargar_modelos():
    paquete = AQUI / "modelos.pkl"
    if paquete.exists():
        with open(paquete, "rb") as archivo:
            datos = pickle.load(archivo)
    else:
        with open(AQUI / "modelo_intencion_mlp.pkl", "rb") as archivo:
            intencion = pickle.load(archivo)
        with open(AQUI / "modelo_sentimiento_mlp.pkl", "rb") as archivo:
            sentimiento = pickle.load(archivo)
        datos = {"intencion": intencion, "sentimiento": sentimiento}

    palabras = datos.get("palabras") or {}
    frases = [tuple(f) for f in datos.get("frases") or []]
    return datos["intencion"], datos["sentimiento"], palabras, frases


MODELO_INTENCION, MODELO_SENTIMIENTO, PALABRAS, FRASES = cargar_modelos()
PATRON = compilar(PALABRAS)

INTENCIONES = list(getattr(MODELO_INTENCION, "classes_", []))
SENTIMIENTOS = list(getattr(MODELO_SENTIMIENTO, "classes_", []))


# ---------------------------------------------------------------------
# Funciones de las pestañas
# ---------------------------------------------------------------------
def clasificar(texto, umbral):
    if not texto or not texto.strip():
        return "", "", 0.0, "", pd.DataFrame(columns=["Intención", "Probabilidad"])

    neutro = normalizar(texto, PALABRAS, FRASES, PATRON)

    prob_intencion = MODELO_INTENCION.predict_proba([neutro])[0]
    prob_sentimiento = MODELO_SENTIMIENTO.predict_proba([neutro])[0]

    indice_intencion = int(prob_intencion.argmax())
    indice_sentimiento = int(prob_sentimiento.argmax())
    intencion = MODELO_INTENCION.classes_[indice_intencion]
    sentimiento = MODELO_SENTIMIENTO.classes_[indice_sentimiento]
    confianza = round(float(prob_intencion[indice_intencion]) * 100, 1)

    aviso = (
        "Confianza suficiente: el mensaje cae dentro de lo que el modelo aprendió."
        if confianza >= umbral
        else f"Confianza baja ({confianza}%): el mensaje queda fuera de lo que vio el modelo. Tomalo como referencia, no como respuesta."
    )

    tabla = pd.DataFrame(
        {
            "Intención": [str(c) for c in MODELO_INTENCION.classes_],
            "Probabilidad": [round(float(p) * 100, 1) for p in prob_intencion],
        }
    ).sort_values("Probabilidad", ascending=False)

    return str(intencion), str(sentimiento), confianza, aviso, tabla


def traducir(texto):
    if not texto or not texto.strip():
        return "", pd.DataFrame(columns=["Expresión", "Se convierte en"])
    neutro = normalizar(texto, PALABRAS, FRASES, PATRON)
    hallazgos, cobertura = buscar_expresiones(texto, PALABRAS, FRASES, PATRON)
    tabla = pd.DataFrame(hallazgos, columns=["Expresión", "Se convierte en"])
    return neutro, tabla


def comparar(neutro_manual, automatico):
    veredicto = (
        "Coinciden palabra por palabra."
        if (neutro_manual or "").strip().lower() == automatico.strip().lower()
        else "No coinciden exactamente: mirá dónde cambia el sentido."
    )
    return automatico, veredicto


def cobertura():
    tabla = pd.DataFrame(
        {
            "Expresión": [p for p in sorted(PALABRAS) if PALABRAS[p].lower() != p.lower()],
            "Neutro": [PALABRAS[p] for p in sorted(PALABRAS) if PALABRAS[p].lower() != p.lower()],
        }
    )
    resumen = (
        f"Diccionario cargado: {len(FRASES)} frases y {len(PALABRAS)} palabras "
        f"({len(FRASES) + len(PALABRAS)} expresiones en total, "
        f"sin contar las que ya estaban en neutro)."
    )
    return resumen, tabla


def traducir_completo(texto):
    """Versión para la pestaña: traduce una sola vez y arma las tres salidas."""
    neutro, tabla = traducir(texto)
    return neutro, resumir(tabla.values.tolist()), tabla


def resumir(expresiones):
    if not expresiones:
        return "No se detectó jerga del diccionario: el mensaje ya estaba en neutro."
    listado = ", ".join(f"«{original}» pasa a «{reemplazo}»" for original, reemplazo in expresiones[:12])
    resto = "" if len(expresiones) <= 12 else f" y {len(expresiones) - 12} más"
    cuantas = "1 expresión" if len(expresiones) == 1 else f"{len(expresiones)} expresiones"
    return f"Se tradujo {cuantas}: {listado}{resto}."


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
titulo = "Evaluación y normalización de textos rioplatenses"
subtitulo = (
    "TP2 de Modelizado de Sistemas de IA. El modelo se entrenó con un corpus sintético "
    "armado por el grupo: sirve para mostrar el método, no para medir precisión real."
)

with gr.Blocks(title=titulo, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {titulo}\n{subtitulo}")

    with gr.Tab("Probá tu mensaje"):
        with gr.Row():
            with gr.Column(scale=3):
                entrada = gr.Textbox(
                    label="Escribí un mensaje como se habla en Argentina",
                    placeholder="Che, re zarpado el laburo, ¿me pasás una mano con la entrega?",
                    lines=3,
                )
                umbral = gr.Slider(0, 100, value=55, step=5, label="Umbral de confianza para avisar")
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column(scale=2):
                salida_intencion = gr.Textbox(label="Intención detectada")
                salida_sentimiento = gr.Textbox(label="Sentimiento detectado")
                salida_confianza = gr.Number(label="Confianza de la intención (%)")
                salida_aviso = gr.Textbox(label="Aviso", lines=2)
        salida_tabla = gr.Dataframe(label="Cómo se repartieron las probabilidades", wrap=True)
        gr.Examples(
            [
                "Che, re zarpado el laburo, ¿me pasás una mano con la entrega?",
                "Estoy al horno con la facu, no llego ni en pedo",
                "xq no me avisaste que había que entregar hoy?",
                "Buenísimo, quedó joya la presentación",
                "Ese pibe es un groso, la rompió toda",
                "Re copado el finde, fuimos a morfar algo",
            ],
            inputs=entrada,
        )
        boton.click(
            clasificar,
            inputs=[entrada, umbral],
            outputs=[salida_intencion, salida_sentimiento, salida_confianza, salida_aviso, salida_tabla],
        )

    with gr.Tab("Argentino vs neutro"):
        gr.Markdown(
            "Compará tu versión en neutro con la que produce el normalizador. "
            "Es el ejercicio que muestra si el diccionario alcanza o si falta jerga."
        )
        with gr.Row():
            texto_original = gr.Textbox(label="Mensaje original", lines=3)
            texto_manual = gr.Textbox(label="Tu versión en neutro", lines=3)
        comparar_boton = gr.Button("Comparar", variant="primary")
        texto_automatico = gr.Textbox(label="Versión automática del normalizador", lines=3)
        veredicto = gr.Textbox(label="Resultado de la comparación")
        comparar_boton.click(
            lambda original, manual: comparar(manual, normalizar(original, PALABRAS, FRASES, PATRON)),
            inputs=[texto_original, texto_manual],
            outputs=[texto_automatico, veredicto],
        )

    with gr.Tab("Traductor de jerga"):
        gr.Markdown("Pegá un mensaje y mirá qué expresiones del diccionario se reconocen.")
        jerga = gr.Textbox(label="Mensaje", lines=4, placeholder="Re copado el finde, fuimos a morfar algo")
        traducir_boton = gr.Button("Traducir", variant="primary")
        jerga_salida = gr.Textbox(label="En español neutro", lines=4)
        jerga_resumen = gr.Textbox(label="Qué se reconoció")
        jerga_tabla = gr.Dataframe(label="Expresiones encontradas", headers=["Expresión", "Se convierte en"])
        traducir_boton.click(
            traducir_completo,
            inputs=jerga,
            outputs=[jerga_salida, jerga_resumen, jerga_tabla],
        )
        gr.Examples(
            [
                "Che, estoy al horno con el laburo, me colgué con la entrega",
                "Estoy al horno con la facu, no llego ni en pedo",
                "Re copado el finde, fuimos a morfar algo",
                "Me da cosa preguntar de nuevo, pero no entendí un jopo",
            ],
            inputs=jerga,
        )

    with gr.Tab("Cobertura del diccionario"):
        gr.Markdown(
            "Todo lo que el normalizador reconoce hoy. Es la parte que crece sin reentrenar el modelo."
        )
        cobertura_boton = gr.Button("Ver diccionario", variant="primary")
        cobertura_resumen = gr.Textbox(label="Resumen")
        cobertura_tabla = gr.Dataframe(label="Expresiones cargadas", wrap=True)
        demo.load(cobertura, outputs=[cobertura_resumen, cobertura_tabla])
        cobertura_boton.click(cobertura, outputs=[cobertura_resumen, cobertura_tabla])

    # Si el grupo sube las figuras o el informe, aparecen solas.
    figuras = [f for f in ["04_matriz_intencion.png", "05_cruce_intencion_sentimiento.png"] if (AQUI / f).exists()]
    if figuras:
        with gr.Tab("Evidencia"):
            gr.Markdown(
                "Matriz de confusión y cruce intención–sentimiento de la corrida. "
                "El sentimiento queda determinado por la intención: el modelo aprende el atajo del corpus."
            )
            for figura in figuras:
                gr.Image(str(AQUI / figura), label=figura)

    if (AQUI / "informe_TP2_evaluacion_textos.pdf").exists():
        with gr.Tab("Informe"):
            gr.File(str(AQUI / "informe_TP2_evaluacion_textos.pdf"), label="Informe del TP2 (PDF)")


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=puerto)