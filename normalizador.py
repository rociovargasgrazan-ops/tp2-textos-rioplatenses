"""Normalización de jerga rioplatense a español neutro — versión del Space.

Misma lógica que el notebook (Paso 8), con los tres blindajes que faltaban:
  1) horarios protegidos (15:30, 7.45),
  2) "La Plata" ciudad protegida por la mayúscula,
  3) usos literales protegidos con testigo ("cuchara de plata", "mina de oro",
     "caña de pescar"): sin esto, el diccionario los convertía a dinero.

El diccionario (palabras y frases) NO está acá: llega dentro de modelos.pkl.
"""

import re

ADJETIVOS = (
    "tarde|temprano|bien|mal|lindo|linda|feo|fea|caro|barato|r[aá]pido|lento|"
    "bueno|buena|rica|rico|copad[oa]s?|zarpad[oa]s?|groso|grosso"
)

# Usos literales: se tapan antes de diccionario y se restauran al final.
# (patrón, cómo tiene que quedar escrito)
LITERALES_PROTEGIDOS = [
    (r"\bmango\s+de\s+la\s+olla\b", "mango de la olla"),
    (r"\bmango\s+del\s+cuchillo\b", "mango del cuchillo"),
    (r"\bcuchara\s+de\s+plata\b", "cuchara de plata"),
    (r"\bmedalla\s+de\s+plata\b", "medalla de plata"),
    (r"\bmoneda\s+de\s+plata\b", "moneda de plata"),
    (r"\bminas\s+de\s+carb[oó]n\b", "minas de carbón"),
    (r"\bmina\s+de\s+carb[oó]n\b", "mina de carbón"),
    (r"\bminas\s+de\s+oro\b", "minas de oro"),
    (r"\bmina\s+de\s+oro\b", "mina de oro"),
    (r"\bchorro\s+de\s+agua\b", "chorro de agua"),
    (r"\btacho\s+de\s+basura\b", "tacho de basura"),
    (r"\bca[ñn]as\s+de\s+pescar\b", "cañas de pescar"),
    (r"\bca[ñn]a\s+de\s+pescar\b", "caña de pescar"),
    (r"\bca[ñn]as\s+de\s+az[uú]car\b", "cañas de azúcar"),
    (r"\bca[ñn]a\s+de\s+az[uú]car\b", "caña de azúcar"),
    (r"\banillo\s+de\s+plata\b", "anillo de plata"),
    (r"\bcadena\s+de\s+plata\b", "cadena de plata"),
    (r"\baros?\s+de\s+plata\b", "aros de plata"),
    (r"\bpulsera\s+de\s+plata\b", "pulsera de plata"),
    (r"\bcubiertos\s+de\s+plata\b", "cubiertos de plata"),
    (r"\bvajilla\s+de\s+plata\b", "vajilla de plata"),
    (r"\bes\s+de\s+plata\b", "es de plata"),
    (r"\bmar\s+del\s+plata\b", "Mar del Plata"),
    (r"\br[ií]o\s+de\s+la\s+plata\b", "Río de la Plata"),
]


def compilar(palabras):
    """Un solo patrón para todo el diccionario: una pasada en vez de cientos."""
    if not palabras:
        return None
    claves = sorted(palabras, key=len, reverse=True)
    patron = r"\b(?:" + "|".join(re.escape(c) for c in claves) + r")\b"
    return re.compile(patron, re.IGNORECASE)


def normalizar(texto, palabras, frases, patron=None, literales=None):
    """Devuelve el mensaje pasado a español neutro."""
    if not texto:
        return ""
    if patron is None:
        patron = compilar(palabras)
    if literales is None:
        literales = LITERALES_PROTEGIDOS

    res = texto.strip()

    # (1) Horarios: no se tocan.
    horarios = []

    def _guardar_hora(coincidencia):
        horarios.append(coincidencia.group(0))
        return f"@@H{len(horarios) - 1}@@"

    res = re.sub(r"\b\d{1,2}[:.]\d{2}\b", _guardar_hora, res)

    # (2) "La Plata" (ciudad) se distingue por la mayúscula: "la plata" es dinero.
    ciudades = []

    def _guardar_ciudad(coincidencia):
        ciudades.append(coincidencia.group(0))
        return f"@@CIU{len(ciudades) - 1}@@"

    res = re.sub(r"\bLa\s+Plata\b", _guardar_ciudad, res)

    # (3) Literales: se tapan para que la pasada de palabras no los rompa.
    guardados = {}
    for i, (patron_lit, texto_lit) in enumerate(literales):
        testigo = f"@@LIT{i}@@"
        nuevo, cuantos = re.subn(patron_lit, testigo, res, flags=re.IGNORECASE)
        if cuantos:
            res = nuevo
            guardados[testigo] = texto_lit

    # (4) "re" solo es "muy" delante de un adjetivo.
    res = re.sub(
        rf"\bre\s+({ADJETIVOS})\b",
        lambda m: f"muy {m.group(1)}",
        res,
        flags=re.IGNORECASE,
    )

    # (5) Primero las frases, que tienen prioridad sobre las palabras sueltas.
    for patron_frase, reemplazo in frases:
        res = re.sub(patron_frase, reemplazo, res, flags=re.IGNORECASE)

    # (6) Después todo el diccionario en una sola pasada.
    if patron is not None:
        res = patron.sub(lambda m: palabras.get(m.group(0).lower(), m.group(0)), res)

    # (7) Se restauran literales y ciudades antes de tocar espacios y mayúsculas.
    for testigo, texto_lit in guardados.items():
        res = res.replace(testigo, texto_lit)
    for i, ciudad in enumerate(ciudades):
        res = res.replace(f"@@CIU{i}@@", ciudad)

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
