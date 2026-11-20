"""Normalización de jerga rioplatense a español neutro.

La lógica es exactamente la del notebook (Paso 8):
  1) protege horarios,
  2) convierte "re + adjetivo" en "muy + adjetivo",
  3) aplica las frases,
  4) aplica las palabras sueltas en una sola pasada.

El diccionario NO está acá: llega dentro de modelos.pkl, exportado desde
la corrida real, así la app usa exactamente el mismo vocabulario del TP.
"""

import re

ADJETIVOS = (
    "tarde|temprano|bien|mal|lindo|linda|feo|fea|caro|barato|r[aá]pido|lento|"
    "bueno|buena|rica|rico|copad[oa]s?|zarpad[oa]s?|groso|grosso"
)


def compilar(palabras):
    """Un solo patrón para todo el diccionario: una pasada en vez de cientos."""
    if not palabras:
        return None
    claves = sorted(palabras, key=len, reverse=True)
    patron = r"\b(?:" + "|".join(re.escape(c) for c in claves) + r")\b"
    return re.compile(patron, re.IGNORECASE)


def normalizar(texto, palabras, frases, patron=None):
    """Devuelve el mensaje pasado a español neutro."""
    if not texto:
        return ""
    if patron is None:
        patron = compilar(palabras)

    res = texto.strip()

    # Horarios protegidos para no romperlos con nada.
    horarios = []

    def _guardar(coincidencia):
        horarios.append(coincidencia.group(0))
        return f"@@H{len(horarios) - 1}@@"

    res = re.sub(r"\b\d{1,2}[:.]\d{2}\b", _guardar, res)

    # "re" solo es "muy" delante de un adjetivo.
    res = re.sub(
        rf"\bre\s+({ADJETIVOS})\b",
        lambda m: f"muy {m.group(1)}",
        res,
        flags=re.IGNORECASE,
    )

    # Primero las frases, que tienen prioridad sobre las palabras sueltas.
    for patron_frase, reemplazo in frases:
        res = re.sub(patron_frase, reemplazo, res, flags=re.IGNORECASE)

    # Después todo el diccionario en una sola pasada.
    if patron is not None:
        res = patron.sub(lambda m: palabras.get(m.group(0).lower(), m.group(0)), res)

    res = re.sub(r"\s+", " ", res).strip()

    # Si arrancaba con "xq" / "pq" era una pregunta: "por qué".
    if re.match(r"^\s*(?:xq|pq)\b", texto, flags=re.IGNORECASE) and res.lower().startswith("porque"):
        res = "Por qué" + res[len("porque"):]

    if res:
        res = res[0].upper() + res[1:]

    for indice, horario in enumerate(horarios):
        res = res.replace(f"@@H{indice}@@", horario)

    return res


def buscar_expresiones(texto, palabras, frases, patron=None):
    """Lista las expresiones del diccionario que aparecen en el texto."""
    if not texto:
        return [], 0
    if patron is None:
        patron = compilar(palabras)

    hallazgos = []
    for patron_frase, reemplazo in frases:
        for coincidencia in re.finditer(patron_frase, texto, flags=re.IGNORECASE):
            if coincidencia.group(0).lower() == str(reemplazo).lower():
                continue  # el diccionario lo deja igual: no vale mostrarlo
            hallazgos.append((coincidencia.group(0), reemplazo, coincidencia.start()))

    if patron is not None:
        for coincidencia in patron.finditer(texto):
            palabra = coincidencia.group(0).lower()
            if palabras.get(palabra, palabra).lower() == palabra:
                continue  # mismo caso: no aporta nada a la tabla
            hallazgos.append(
                (coincidencia.group(0), palabras.get(palabra, palabra), coincidencia.start())
            )

    # Si dos coincidencias se pisan, gana la más larga.
    usados, limpias = [], []
    for hallado, reemplazo, posicion in sorted(hallazgos, key=lambda x: (-len(x[0]), x[2])):
        if any(posicion < fin and inicio < posicion + len(hallado) for inicio, fin in usados):
            continue
        usados.append((posicion, posicion + len(hallado)))
        limpias.append((hallado, reemplazo))

    total_palabras = max(len(texto.split()), 1)
    cubiertas = sum(len(hallado.split()) for hallado, _ in limpias)
    return limpias, round(100 * cubiertas / total_palabras)
