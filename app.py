"""App del TP2 — Evaluación y normalización de textos rioplatenses (5 pestañas).

Space de Gradio en Python. Necesita en la misma carpeta:
    app.py                 (este archivo)
    normalizador.py        (reglas de normalización)
    modelos.pkl            (diccionario + los dos MLP entrenados en el notebook)
    requirements.txt

Si falta modelos.pkl, la app igual abre: las pestañas 2, 3 y 5 funcionan con el
JSON del diccionario, y la pestaña 1 avisa que no encontró los modelos.
"""

import json
import pickle
import re
from pathlib import Path

import gradio as gr
import pandas as pd

from normalizador import LITERALES_PROTEGIDOS, buscar_expresiones, compilar, normalizar

AQUI = Path(__file__).parent


# ---------------------------------------------------------------------
# 1. Carga del paquete: modelos + diccionario
# ---------------------------------------------------------------------
def cargar_paquete():
    """Devuelve (modelo_intencion, modelo_sentimiento, palabras, frases, origen, error)."""
    paquete = AQUI / "modelos.pkl"
    datos = {}
    error_pkl = ""

    try:
        if paquete.exists():
            with open(paquete, "rb") as archivo:
                datos = pickle.load(archivo)
            origen = "modelos.pkl"
        elif (AQUI / "modelo_intencion_mlp.pkl").exists():
            with open(AQUI / "modelo_intencion_mlp.pkl", "rb") as archivo:
                datos["intencion"] = pickle.load(archivo)
            with open(AQUI / "modelo_sentimiento_mlp.pkl", "rb") as archivo:
                datos["sentimiento"] = pickle.load(archivo)
            origen = "los dos .pkl separados"
        else:
            origen = "sin modelos (solo diccionario)"
    except Exception as error:  # pkl de otra versión de scikit-learn, archivo cortado, etc.
        print(f"No se pudo abrir el paquete de modelos: {error}")
        datos = {}
        origen = f"sin modelos ({type(error).__name__})"
        error_pkl = f"{type(error).__name__}: {error}"

    palabras = datos.get("palabras") or {}
    frases = [tuple(f) for f in (datos.get("frases") or [])]

    # Si el pkl no trajo diccionario, se busca el JSON que exporta el notebook.
    if not palabras:
        for nombre in ("diccionario_actualizado.json", "diccionario.json"):
            ruta = AQUI / nombre
            if ruta.exists():
                with open(ruta, encoding="utf-8") as archivo:
                    plan = json.load(archivo)
                palabras = plan.get("palabras") or {}
                frases = [tuple(f) for f in (plan.get("frases") or [])]
                origen = f"{origen} + {nombre}"
                break

    return datos.get("intencion"), datos.get("sentimiento"), palabras, frases, origen, error_pkl


MODELO_INTENCION, MODELO_SENTIMIENTO, PALABRAS, FRASES, ORIGEN, ERROR_PKL = cargar_paquete()


# ---------------------------------------------------------------------
# 1 bis. Formas conjugadas: el diccionario trae "morfar", no "morfaron"
# ---------------------------------------------------------------------
# Se generan solo para verbos regulares y con reemplazo de una sola palabra.
# No se pisan formas que ya estén cargadas en el diccionario.
CONJUGACIONES = [
    ("o", "o", "o"), ("ás", "és", "ís"), ("a", "e", "e"),
    ("amos", "emos", "imos"), ("áis", "éis", "ís"), ("an", "en", "en"),
    ("é", "í", "í"), ("aste", "iste", "iste"), ("ó", "ió", "ió"),
    ("amos", "imos", "imos"), ("asteis", "isteis", "isteis"), ("aron", "ieron", "ieron"),
    ("aba", "ía", "ía"), ("abas", "ías", "ías"), ("aba", "ía", "ía"),
    ("ábamos", "íamos", "íamos"), ("abais", "íais", "íais"), ("aban", "ían", "ían"),
    ("aré", "eré", "iré"), ("arás", "erás", "irás"), ("ará", "erá", "irá"),
    ("aremos", "eremos", "iremos"), ("aréis", "eréis", "iréis"), ("arán", "erán", "irán"),
    ("ando", "iendo", "iendo"), ("ado", "ido", "ido"),
]
CLASE_VERBO = {"ar": 0, "er": 1, "ir": 2}
LETRAS_QUE_CAMBIAN = ("c", "z", "g", "q")  # arranqué, empecé: necesitan otra ortografía


def _formas_conjugadas(palabras):
    """Devuelve {forma conjugada: reemplazo} para los verbos regulares del diccionario."""
    formas = {}
    for infinitivo, reemplazo in palabras.items():
        if not re.fullmatch(r"[a-záéíóúñ]{3,}(ar|er|ir)", infinitivo):
            continue
        traduccion = str(reemplazo).strip().lower()
        if not re.fullmatch(r"[a-záéíóúñ]{3,}(ar|er|ir)", traduccion):
            continue
        if " " in traduccion:
            continue

        clase_origen = CLASE_VERBO[infinitivo[-2:]]
        clase_destino = CLASE_VERBO[traduccion[-2:]]
        raiz_origen = infinitivo[:-2]
        raiz_destino = traduccion[:-2]
        if raiz_origen.endswith(LETRAS_QUE_CAMBIAN) or raiz_destino.endswith(LETRAS_QUE_CAMBIAN):
            continue
        if raiz_origen == raiz_destino:
            continue

        for terminaciones in CONJUGACIONES:
            forma = raiz_origen + terminaciones[clase_origen]
            nueva = raiz_destino + terminaciones[clase_destino]
            if forma not in palabras:
                formas.setdefault(forma, nueva)
    return formas


FORMAS_VERBALES = _formas_conjugadas(PALABRAS)
for _forma, _traduccion in FORMAS_VERBALES.items():
    PALABRAS.setdefault(_forma, _traduccion)
if FORMAS_VERBALES:
    print(f"Formas conjugadas agregadas: {len(FORMAS_VERBALES)}.")

PATRON = compilar(PALABRAS)
HAY_MODELOS = MODELO_INTENCION is not None and MODELO_SENTIMIENTO is not None

print(
    f"Cargado: {len(PALABRAS)} palabras y {len(FRASES)} frases desde {ORIGEN}. "
    f"Modelos: {'sí' if HAY_MODELOS else 'no'}."
)


# ---------------------------------------------------------------------
# 2. Pestaña 1 — probá tu mensaje (intención + sentimiento)
# ---------------------------------------------------------------------
def clasificar(texto, umbral):
    vacio = pd.DataFrame(columns=["Intención", "Probabilidad"])

    if not texto or not texto.strip():
        return "", "", 0.0, "", vacio

    if not HAY_MODELOS:
        return (
            "Sin modelos",
            "Sin modelos",
            0.0,
            "No pude usar los modelos de la pestaña 1. "
            + (
                f"El archivo modelos.pkl está pero no se pudo abrir ({ERROR_PKL}). "
                "Suele ser que scikit-learn no es la misma versión con la que se generó: "
                "fijala en requirements.txt y reiniciá el Space. "
                if ERROR_PKL
                else "Falta modelos.pkl en el Space: subilo y reiniciá la app. "
            )
            + "Mientras tanto, las pestañas 2, 3 y 5 funcionan igual.",
            vacio,
        )

    neutro = normalizar(texto, PALABRAS, FRASES, PATRON)

    prob_intencion = MODELO_INTENCION.predict_proba([neutro])[0]
    prob_sentimiento = MODELO_SENTIMIENTO.predict_proba([neutro])[0]

    indice_intencion = int(prob_intencion.argmax())
    indice_sentimiento = int(prob_sentimiento.argmax())
    intencion = MODELO_INTENCION.classes_[indice_intencion]
    sentimiento = MODELO_SENTIMIENTO.classes_[indice_sentimiento]
    confianza = round(float(prob_intencion[indice_intencion]) * 100, 1)

    if confianza >= umbral:
        aviso = (
            "Confianza suficiente: el mensaje cae dentro de lo que el modelo aprendió. "
            "Igual el corpus es sintético: sirve para mostrar el método, no como medida de precisión real."
        )
    else:
        aviso = (
            f"Confianza baja ({confianza}%): el mensaje queda fuera de lo que vio el modelo. "
            "Tomalo como referencia, no como respuesta."
        )

    tabla = pd.DataFrame(
        {
            "Intención": [str(c) for c in MODELO_INTENCION.classes_],
            "Probabilidad": [round(float(p) * 100, 1) for p in prob_intencion],
        }
    ).sort_values("Probabilidad", ascending=False)

    return str(intencion), str(sentimiento), confianza, aviso, tabla


# ---------------------------------------------------------------------
# 3. Pestañas 2 y 3 — normalización y comparación
# ---------------------------------------------------------------------
def comparar(manual, automatico):
    if not (manual or "").strip():
        return automatico, "Escribí tu versión en neutro para poder comparar."
    if (manual or "").strip().lower() == automatico.strip().lower():
        return automatico, "Coinciden palabra por palabra."
    return automatico, "No coinciden exactamente: mirá dónde cambia el sentido."


def resumir(expresiones):
    if not expresiones:
        return "No se detectó jerga del diccionario: el mensaje ya estaba en neutro."
    listado = ", ".join(
        f"«{original}» pasa a «{reemplazo}»" for original, reemplazo in expresiones[:12]
    )
    resto = "" if len(expresiones) <= 12 else f" y {len(expresiones) - 12} más"
    cuantas = "1 expresión" if len(expresiones) == 1 else f"{len(expresiones)} expresiones"
    return f"Se tradujo {cuantas}: {listado}{resto}."


def _sin_protegidos(texto):
    """Tapa lo que el normalizador deja intacto (horarios, "La Plata", literales).

    Sirve solo para la tabla de la pestaña 3: si no, la tabla dice que "La Plata"
    se convirtió en "dinero" aunque la frase quedó bien escrita.
    """
    res = re.sub(r"\b\d{1,2}[:.]\d{2}\b", " __HORA__ ", texto)
    res = re.sub(r"\bLa\s+Plata\b", " __CIUDAD__ ", res)
    for patron_lit, _ in LITERALES_PROTEGIDOS:
        res = re.sub(patron_lit, " __LITERAL__ ", res, flags=re.IGNORECASE)
    return res


def traducir_completo(texto):
    columnas = pd.DataFrame(columns=["Expresión", "Se convierte en"])
    if not texto or not texto.strip():
        return "", "Escribí un mensaje.", columnas

    neutro = normalizar(texto, PALABRAS, FRASES, PATRON)
    hallazgos, _ = buscar_expresiones(_sin_protegidos(texto), PALABRAS, FRASES, PATRON)
    tabla = pd.DataFrame(hallazgos, columns=["Expresión", "Se convierte en"])
    return neutro, resumir(hallazgos), tabla


def cobertura():
    claves = [p for p in sorted(PALABRAS) if PALABRAS[p].lower() != p.lower()]
    tabla = pd.DataFrame(
        {"Expresión": claves, "Neutro": [PALABRAS[p] for p in claves]}
    )
    resumen = (
        f"Diccionario cargado: {len(FRASES)} frases y {len(PALABRAS)} palabras "
        f"({len(FRASES) + len(PALABRAS)} expresiones en total). Origen: {ORIGEN}."
    )
    return resumen, tabla


# ---------------------------------------------------------------------
# 4. Pestaña 5 — revisión de ambigüedad
# ---------------------------------------------------------------------
OPCION_AUTO = "Que decida el sistema"
OPCION_JERGA = "Leerlo como jerga"
OPCION_LITERAL = "Leerlo como literal"
OPCIONES_AMBIGUEDAD = [OPCION_AUTO, OPCION_JERGA, OPCION_LITERAL]

TERMINOS_AMBIGUOS = {
    "mango": {
        "patron": r"\bmangos?\b",
        "sentido_jerga": "dinero",
        "sentido_literal": "la fruta",
        "palabra_jerga": "dinero",
        "reglas_jerga": [
            (r"\bno\s+tengo\s+un\s+mango\b", "no tengo nada de dinero"),
            (r"\bsin\s+un\s+mango\b", "sin nada de dinero"),
            (r"\bdos\s+mangos\b", "muy poco dinero"),
            (r"\bun\s+mango\b", "dinero"),
            (r"\bmangos?\b", "dinero"),
        ],
    },
    "cana": {
        "patron": r"\bcanas?\b",
        "sentido_jerga": "policía o cárcel",
        "sentido_literal": "la caña de pescar",
        "palabra_jerga": "polic",
        "reglas_jerga": [
            (r"\bla\s+cana\b", "la policía"),
            (r"\buna\s+cana\b", "una cerveza"),
            (r"\bcanas?\b", "la policía"),
        ],
    },
    "plata": {
        "patron": r"\bplatas?\b",
        "sentido_jerga": "dinero",
        "sentido_literal": "el metal o la ciudad de La Plata",
        "palabra_jerga": "dinero",
        # "La Plata" (la ciudad) no se toca ni cuando se fuerza la jerga.
        "protegidos": [r"\bLa\s+Plata\b"],
        "reglas_jerga": [
            (r"\ba\s+la\s+plata\b", "al dinero"),
            (r"\bde\s+la\s+plata\b", "del dinero"),
            (r"\bla\s+plata\b", "el dinero"),
            (r"\buna\s+plata\b", "dinero"),
            (r"\bplatas?\b", "dinero"),
        ],
    },
    "gamba": {
        "patron": r"\bgambas?\b",
        "sentido_jerga": "cien pesos",
        "sentido_literal": "el langostino",
        "palabra_jerga": "cien pesos",
        "reglas_jerga": [
            (r"\buna\s+gamba\b", "cien pesos"),
            (r"\bgambas?\b", "cien pesos"),
        ],
    },
}


def _escapar(texto):
    return texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tapar_termino(texto, patron, prefijo):
    guardados = []

    def _guardar(m):
        guardados.append(m.group(0))
        return f"@@{prefijo}{len(guardados) - 1}@@"

    return re.sub(patron, _guardar, texto, flags=re.IGNORECASE), guardados


def _tapar_varios(texto, patrones, prefijo):
    guardados = []

    def _guardar(m):
        guardados.append(m.group(0))
        return f"@@{prefijo}{len(guardados) - 1}@@"

    for patron in patrones:
        # Case-sensitive a propósito: "La Plata" ciudad se distingue por la mayúscula.
        texto = re.sub(patron, _guardar, texto)
    return texto, guardados


def _destapar(texto, guardados, prefijo):
    for i, palabra in enumerate(guardados):
        texto = texto.replace(f"@@{prefijo}{i}@@", palabra)
    return texto


def _fragmento(texto, patron, ancho=32):
    m = re.search(patron, texto, flags=re.IGNORECASE)
    if not m:
        return ""
    ini = max(0, m.start() - ancho)
    fin = min(len(texto), m.end() + ancho)
    return ("…" if ini else "") + texto[ini:fin].strip() + ("…" if fin < len(texto) else "")


def _neutro(texto):
    return normalizar(texto, PALABRAS, FRASES, PATRON)


def lectura_jerga(texto, clave):
    """Fuerza la lectura de jerga y devuelve el mensaje en neutro."""
    info = TERMINOS_AMBIGUOS[clave]
    prefijo = f"{clave.upper()}CIU"
    trabajo, guardados = _tapar_varios(texto, info.get("protegidos", []), prefijo)
    for patron, reemplazo in info["reglas_jerga"]:
        trabajo = re.sub(patron, reemplazo, trabajo, flags=re.IGNORECASE)
    return _destapar(_neutro(trabajo), guardados, prefijo)


def lectura_literal(texto, clave):
    """Fuerza la lectura literal: el término se queda tal cual está."""
    tapado, guardados = _tapar_termino(texto, TERMINOS_AMBIGUOS[clave]["patron"], clave.upper())
    return _destapar(_neutro(tapado), guardados, clave.upper())


def pantalla_5_ambiguedad(texto, d_mango, d_cana, d_plata, d_gamba):
    """Muestra las dos lecturas de cada término y aplica la corrección humana."""
    if not texto or not texto.strip():
        return "<p>Escribí un mensaje.</p>", "", "", ""

    decisiones = dict(zip(TERMINOS_AMBIGUOS, [d_mango, d_cana, d_plata, d_gamba]))

    # Lectura automática: la del sistema, sin intervención.
    automatico = _neutro(texto)

    presentes = [
        clave
        for clave in TERMINOS_AMBIGUOS
        if re.search(TERMINOS_AMBIGUOS[clave]["patron"], texto, flags=re.IGNORECASE)
    ]
    faltantes = [clave for clave in TERMINOS_AMBIGUOS if clave not in presentes]

    # --- Si el mensaje no trae ninguna de las cuatro palabras ---
    if not presentes:
        hallazgos, _ = buscar_expresiones(_sin_protegidos(texto), PALABRAS, FRASES, PATRON)
        if hallazgos:
            lista = "".join(
                f"<li>«{_escapar(original)}» → «{_escapar(str(reemplazo))}»</li>"
                for original, reemplazo in hallazgos[:8]
            )
            detalle = (
                "<p style='margin:8px 0 4px'>Lo que sí reconoció el diccionario en tu mensaje:</p>"
                f"<ul style='margin:0 0 8px 18px'>{lista}</ul>"
            )
        else:
            detalle = (
                "<p style='margin:8px 0'>Tampoco hay jerga cargada en tu mensaje: "
                "quedó casi igual, palabra por palabra.</p>"
            )
        tarjeta = (
            "<div style='border:1px solid #e5e7eb; border-radius:10px; padding:12px 14px; "
            "background:#f9fafb;'>"
            "<b>Tu mensaje no trae ninguna de las cuatro palabras</b> "
            "(mango, cana, plata, gamba), así que en esta pestaña no hay ambigüedad para revisar. "
            "Ojo: eso no quiere decir que no se haya analizado; abajo está la lectura del sistema."
            f"{detalle}"
            "<p style='margin:8px 0 0; color:#6b7280; font-size:13px'>"
            "Probá con: «No tengo un mango, la cana me pidió plata y encima me comí una gamba»."
            "</p></div>"
        )
        return (
            tarjeta,
            automatico,
            automatico,
            "No hay términos ambiguos en el mensaje: no hubo nada que corregir.",
        )

    # Lectura con las decisiones humanas: se tapa lo que la persona marcó.
    trabajo = texto
    tapados = []
    for clave, decision in decisiones.items():
        info = TERMINOS_AMBIGUOS[clave]
        prefijo = f"{clave.upper()}INT"
        if decision == OPCION_LITERAL:
            trabajo, guardados = _tapar_termino(trabajo, info["patron"], prefijo)
            tapados.append((prefijo, guardados))
        elif decision == OPCION_JERGA:
            for patron, reemplazo in info["reglas_jerga"]:
                trabajo = re.sub(patron, reemplazo, trabajo, flags=re.IGNORECASE)

    final = _neutro(trabajo)
    for prefijo, guardados in tapados:
        final = _destapar(final, guardados, prefijo)

    # --- Tarjetas: qué leyó el sistema y qué pasaría con cada lectura ---
    tarjetas = []
    for clave in presentes:
        info = TERMINOS_AMBIGUOS[clave]
        patron = info["patron"]
        decision = decisiones[clave]

        quedan = len(re.findall(patron, automatico, flags=re.IGNORECASE))
        aparecen = len(re.findall(patron, texto, flags=re.IGNORECASE))
        dice_jerga = info["palabra_jerga"] in automatico.lower()
        if quedan == 0 and dice_jerga:
            sistema = "jerga"
        elif quedan == aparecen:
            sistema = "literal"
        elif quedan == 0:
            sistema = "lo reescribió"
        else:
            sistema = f"mixto ({quedan} de {aparecen} literal)"

        fr_jerga = lectura_jerga(texto, clave)
        fr_literal = lectura_literal(texto, clave)
        color_j = "#166534" if decision == OPCION_JERGA else "#6b7280"
        color_l = "#166534" if decision == OPCION_LITERAL else "#6b7280"
        fondo_j = "#dcfce7" if decision == OPCION_JERGA else "#ffffff"
        fondo_l = "#dcfce7" if decision == OPCION_LITERAL else "#ffffff"

        tarjetas.append(
            '<div style="border:1px solid #e5e7eb; border-radius:10px; '
            'padding:10px 12px; margin-bottom:8px; background:#f9fafb;">'
            f'<div style="margin-bottom:4px;"><b>{clave}</b> '
            f'<span style="color:#6b7280;">en «{_escapar(_fragmento(texto, patron))}»'
            f' · el sistema leyó: <b>{sistema}</b></span></div>'
            f'<div style="background:{fondo_j}; color:{color_j}; padding:6px 8px; '
            f'border-radius:6px; margin-bottom:4px;">'
            f'<b>Jerga</b> ({info["sentido_jerga"]}): {_escapar(fr_jerga)}</div>'
            f'<div style="background:{fondo_l}; color:{color_l}; padding:6px 8px; '
            f'border-radius:6px;">'
            f'<b>Literal</b> ({info["sentido_literal"]}): {_escapar(fr_literal)}</div>'
            '</div>'
        )

    if faltantes:
        tarjetas.append(
            '<p style="color:#6b7280; font-size:13px; margin:4px 0 0;">'
            f"No aparecen en el mensaje: {', '.join(faltantes)}.</p>"
        )

    cambios = [
        f"{clave} → {'jerga' if d == OPCION_JERGA else 'literal'}"
        for clave, d in decisiones.items()
        if d != OPCION_AUTO
    ]
    if cambios:
        nota = "Corregiste a mano: " + ", ".join(cambios) + "."
    else:
        nota = "Sin correcciones: el texto final es la lectura automática del sistema."

    return "".join(tarjetas), automatico, final, nota


# ---------------------------------------------------------------------
# 4 bis. Las cuatro pantallas de la demo (como el esquema de la PPT)
# ---------------------------------------------------------------------
def _predecir(texto):
    """Devuelve (intención, sentimiento, confianza) o None si no hay modelos."""
    if not HAY_MODELOS:
        return None
    prob_intencion = MODELO_INTENCION.predict_proba([texto])[0]
    prob_sentimiento = MODELO_SENTIMIENTO.predict_proba([texto])[0]
    i = int(prob_intencion.argmax())
    j = int(prob_sentimiento.argmax())
    return (
        str(MODELO_INTENCION.classes_[i]),
        str(MODELO_SENTIMIENTO.classes_[j]),
        round(float(prob_intencion[i]) * 100, 1),
    )


def _jerga_detectada(texto, tope=6):
    hallazgos, _ = buscar_expresiones(_sin_protegidos(texto or ""), PALABRAS, FRASES, PATRON)
    if not hallazgos:
        return "ninguna expresión del diccionario"
    listado = ", ".join(f"«{original}» → «{reemplazo}»" for original, reemplazo in hallazgos[:tope])
    resto = "" if len(hallazgos) <= tope else f" y {len(hallazgos) - tope} más"
    return f"{len(hallazgos)} expresión(es): {listado}{resto}"


def pantalla_1(texto, umbral):
    """1. Probá tu mensaje: intención, sentimiento, confianza y análisis semántico."""
    intencion, sentimiento, confianza, aviso, tabla = clasificar(texto, umbral)

    if not texto or not texto.strip():
        return "", "", 0.0, "", "", tabla

    neutro = _neutro(texto)
    jerga = _jerga_detectada(texto)

    if HAY_MODELOS:
        analisis = (
            f"Tema: {intencion.lower()}. Tono: {sentimiento.lower()}. "
            f"Confianza en la intención: {confianza}%.\n"
            f"Cómo lo lee el normalizador: {neutro}\n"
            f"Jerga reconocida: {jerga}.\n"
            "(Análisis por reglas: este Space no llama a ninguna API.)"
        )
    else:
        analisis = (
            f"Cómo lo lee el normalizador: {neutro}\n"
            f"Jerga reconocida: {jerga}.\n"
            "(Sin modelos cargados no hay clasificación; el resto del análisis sí funciona.)"
        )

    return intencion, sentimiento, confianza, analisis, aviso, tabla


def pantalla_2(texto):
    """2. Argentino vs. neutro: compara las etiquetas de las dos variantes."""
    if not texto or not texto.strip():
        return "", "—", "—", "Escribí un mensaje para comparar."

    neutro = _neutro(texto)

    if not HAY_MODELOS:
        return (
            neutro,
            "Sin modelos",
            "Sin modelos",
            "No encontré modelos.pkl, así que no puedo comparar etiquetas. "
            "La versión en neutro sí está arriba.",
        )

    original = _predecir(texto)
    normalizado = _predecir(neutro)

    etiqueta_original = f"{original[0]} · {original[1]} ({original[2]}%)"
    etiqueta_neutra = f"{normalizado[0]} · {normalizado[1]} ({normalizado[2]}%)"

    if original[0] == normalizado[0] and original[1] == normalizado[1]:
        alerta = (
            "Las dos variantes dan la misma etiqueta: normalizar la jerga no cambió "
            "la lectura del modelo."
        )
    else:
        cambios = []
        if original[0] != normalizado[0]:
            cambios.append(f"intención ({original[0]} → {normalizado[0]})")
        if original[1] != normalizado[1]:
            cambios.append(f"sentimiento ({original[1]} → {normalizado[1]})")
        alerta = "⚠ Cambió la etiqueta al normalizar: " + " y ".join(cambios) + "."

    return neutro, etiqueta_original, etiqueta_neutra, alerta


def pantalla_3(texto):
    """3. Traductor de jerga: normaliza el mensaje y lo vuelve a clasificar."""
    neutro, resumen, tabla = traducir_completo(texto)

    if not texto or not texto.strip():
        return neutro, resumen, tabla, "—"

    if not HAY_MODELOS:
        return neutro, resumen, tabla, "Sin modelos.pkl: no puedo reclasificar."

    intencion, sentimiento, confianza = _predecir(neutro)
    etiqueta = f"{intencion} · {sentimiento} ({confianza}%) sobre el texto en neutro"
    return neutro, resumen, tabla, etiqueta


def tabla_f1():
    return pd.DataFrame(
        {
            "Variante": ["Rioplatense", "Neutro"],
            "F1 ponderado": [1.000, 1.000],
            "Accuracy": [1.000, 1.000],
        }
    )


# ---------------------------------------------------------------------
# 5. Interfaz
# ---------------------------------------------------------------------
FIGURAS = [
    ("04_matriz_intencion.png", "Matriz de confusión · intención"),
    ("05_cruce_intencion_sentimiento.png", "Intención × sentimiento"),
]
FIGURAS = [(nombre, leyenda) for nombre, leyenda in FIGURAS if (AQUI / nombre).exists()]

with gr.Blocks(title="Evaluación de Textos AI — TP2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇦🇷 Evaluación y normalización de textos rioplatenses")
    gr.Markdown(
        "TP2 de Modelizado de Sistemas de IA. El modelo se entrenó con un corpus "
        "sintético armado por el grupo: sirve para mostrar el método, no para medir "
        "precisión real."
    )

    with gr.Tabs():
        # ---------------- 1 ----------------
        with gr.Tab("1. Probá tu mensaje"):
            gr.Markdown(
                "Intención, sentimiento, confianza y análisis semántico. "
                "Las dos redes corren locales: si no hay modelos, la app avisa y sigue."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    txt_1 = gr.Textbox(
                        lines=3,
                        label="Escribí un mensaje como se habla en Argentina",
                        placeholder="Che, hace 4 días que espero la guita y nadie me da bola...",
                    )
                    umbral = gr.Slider(0, 100, value=55, step=5, label="Umbral de confianza para avisar")
                    btn_1 = gr.Button("Analizar mensaje", variant="primary")
                with gr.Column(scale=2):
                    sal_intencion = gr.Textbox(label="Intención detectada")
                    sal_sentimiento = gr.Textbox(label="Sentimiento detectado")
                    sal_confianza = gr.Number(label="Confianza de la intención (%)")
                    sal_aviso = gr.Textbox(label="Aviso", lines=3)
            sal_analisis = gr.Textbox(label="Análisis semántico", lines=5)
            sal_tabla1 = gr.Dataframe(label="Cómo se repartieron las probabilidades", wrap=True)
            gr.Examples(
                [
                    "Che, re zarpado el laburo, ¿me pasás una mano con la entrega?",
                    "Estoy al horno con la facu, no llego ni en pedo",
                    "xq no me avisaste que había que entregar hoy?",
                    "Buenísimo, quedó joya la presentación",
                    "Ese pibe es un groso, la rompió toda",
                    "Re copado el finde, fuimos a morfar algo",
                ],
                inputs=txt_1,
            )
            btn_1.click(
                pantalla_1,
                inputs=[txt_1, umbral],
                outputs=[sal_intencion, sal_sentimiento, sal_confianza, sal_analisis, sal_aviso, sal_tabla1],
            )

        # ---------------- 2 ----------------
        with gr.Tab("2. Argentino vs. neutro"):
            gr.Markdown(
                "Compara las etiquetas de las dos variantes y avisa si cambian: "
                "así se mide si la jerga mueve la lectura del modelo."
            )
            txt_2 = gr.Textbox(
                lines=3,
                label="Mensaje en argentino",
                placeholder="Che, estoy al horno con el laburo, me colgué con la entrega",
            )
            btn_2 = gr.Button("Comparar variantes", variant="primary")
            sal_neutro2 = gr.Textbox(label="Versión en neutro", lines=3)
            with gr.Row():
                sal_et_orig = gr.Textbox(label="Etiqueta · original")
                sal_et_neutro = gr.Textbox(label="Etiqueta · en neutro")
            sal_alerta2 = gr.Textbox(label="Alerta")
            gr.Examples(
                [
                    "Che, estoy al horno con el laburo, me colgué con la entrega",
                    "Estoy al horno con la facu, no llego ni en pedo",
                    "Re copado el finde, fuimos a morfar algo",
                    "Los pibes se morfaron todo y no pusieron ni un peso",
                ],
                inputs=txt_2,
            )
            btn_2.click(
                pantalla_2,
                inputs=txt_2,
                outputs=[sal_neutro2, sal_et_orig, sal_et_neutro, sal_alerta2],
            )

        # ---------------- 3 ----------------
        with gr.Tab("3. Traductor de jerga"):
            gr.Markdown(
                "Normaliza el mensaje y lo vuelve a clasificar, para ver qué queda "
                "después de sacar la jerga."
            )
            txt_3 = gr.Textbox(
                lines=4,
                label="Mensaje",
                placeholder="Re copado el finde, fuimos a morfar algo",
            )
            btn_3 = gr.Button("Traducir y clasificar", variant="primary")
            sal_neutro3 = gr.Textbox(label="En español neutro", lines=4)
            sal_etiqueta3 = gr.Textbox(label="Clasificación del texto en neutro")
            sal_resumen3 = gr.Textbox(label="Qué se reconoció")
            sal_tabla3 = gr.Dataframe(label="Expresiones encontradas", headers=["Expresión", "Se convierte en"])
            gr.Examples(
                [
                    "Che, estoy al horno con el laburo, me colgué con la entrega",
                    "Me da cosa preguntar de nuevo, pero no entendí un jopo",
                    "Los pibes se morfaron todo y no pusieron ni un peso",
                    "Vivo en La Plata y se me rompió la caña de pescar a las 15:30",
                ],
                inputs=txt_3,
            )
            btn_3.click(
                pantalla_3,
                inputs=txt_3,
                outputs=[sal_neutro3, sal_resumen3, sal_tabla3, sal_etiqueta3],
            )

            with gr.Accordion("Revisión fina: ¿jerga o literal? (mango, cana, plata, gamba)", open=False):
                gr.Markdown(
                    "Cuatro palabras que significan dos cosas distintas. El sistema elige "
                    "una lectura según el contexto y acá la podés revisar."
                )
                txt_5 = gr.Textbox(
                    lines=2,
                    label="Mensaje",
                    value="No tengo un mango, la cana me pidió plata y encima me comí una gamba",
                )
                selects_amb = []
                for clave, info in TERMINOS_AMBIGUOS.items():
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown(
                                f"**{clave}** · jerga: {info['sentido_jerga']} · "
                                f"literal: {info['sentido_literal']}"
                            )
                        with gr.Column(scale=2):
                            selects_amb.append(
                                gr.Dropdown(
                                    choices=OPCIONES_AMBIGUEDAD,
                                    value=OPCION_AUTO,
                                    label="Cómo leerlo",
                                )
                            )
                btn_5 = gr.Button("Revisar ambigüedad", variant="secondary")
                sal_tarjetas = gr.HTML()
                with gr.Row():
                    sal_auto5 = gr.Textbox(label="1. Lectura automática del sistema", lines=2)
                    sal_final5 = gr.Textbox(label="2. Lectura con tus decisiones", lines=2)
                sal_nota5 = gr.Textbox(label="Qué corregiste")
                btn_5.click(
                    pantalla_5_ambiguedad,
                    inputs=[txt_5] + selects_amb,
                    outputs=[sal_tarjetas, sal_auto5, sal_final5, sal_nota5],
                )

        # ---------------- 4 ----------------
        with gr.Tab("4. Métricas"):
            gr.Markdown(
                "F1 por variante, matriz de confusión y análisis de lunfardo. "
                "Los números salen de la corrida del notebook, no se calculan acá."
            )
            tabla_f1_ui = gr.Dataframe(
                value=tabla_f1(),
                label="F1 por variante (conjunto de prueba)",
                wrap=True,
            )
            if FIGURAS:
                with gr.Row():
                    for nombre, leyenda in FIGURAS:
                        gr.Image(str(AQUI / nombre), label=leyenda)
            else:
                gr.Markdown(
                    "_Subí `04_matriz_intencion.png` y `05_cruce_intencion_sentimiento.png` "
                    "a este Space y las figuras aparecen acá solas._"
                )
            gr.Markdown(
                "**Análisis de lunfardo.** Solo 112 de los 1.422 mensajes rioplatenses "
                "(7,9%) tienen jerga real, y en prueba quedaron 15 casos. Con 15 casos no "
                "se puede medir ninguna brecha: es un problema de potencia estadística, "
                "no un resultado.\n\n"
                "**Y el 100%.** La matriz no tiene un solo error y cada intención tiene un "
                "único sentimiento asociado: la red de sentimiento no aprendió emociones, "
                "aprendió la etiqueta que ya venía con la intención. El 100% mide que la "
                "tarea no tiene dificultad, no la capacidad del modelo."
            )
            with gr.Accordion("Diccionario cargado (crece sin reentrenar)", open=False):
                resumen_cob = gr.Textbox(label="Resumen")
                tabla_cob = gr.Dataframe(label="Expresiones cargadas", wrap=True)
                demo.load(cobertura, outputs=[resumen_cob, tabla_cob])

if __name__ == "__main__":
    demo.launch()
