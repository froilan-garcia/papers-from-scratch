# ClaudeLarp — Reviews de literatura + implementación de papers

Espacio de trabajo de Froilán (graduado en Física y Matemáticas, Univ. de Oviedo; máster en Big Data Analytics en la UC3M desde septiembre de 2026) para leer papers científicos, escribir reviews estructuradas e implementar su contenido en Python.

## Idioma

- Conversación, reviews y los `README.md` de implementación: **español**.
- Código, nombres de variables y comentarios: **inglés** (convención estándar).
- **`DERIVATIONS.md`: inglés.** Es el único artefacto del repo pensado para enseñarse fuera (solicitudes, entrevistas), así que se escribe en inglés desde el principio en vez de traducirse después.
- Términos técnicos sin traducción asentada se dejan en inglés (p. ej. *embedding*, *gauge*).

*Excepción histórica:* `1996-tibshirani-lasso` tiene su desarrollo en español y se llama `DEDUCCIONES.md`. Se dejó así a propósito; no hay que renombrarlo ni traducirlo salvo que se pida.

## Estructura

```
papers/            PDFs originales. Nombre: <año>-<primer-autor>-<titulo-corto>.pdf
                   Los .pdf NO se versionan (peso + copyright: el repo es público).
                   El inventario con enlaces de descarga es papers/INDEX.md, que sí se versiona.
reviews/           Una review markdown por paper, mismo slug que el PDF.
implementations/   Una carpeta por paper, mismo slug. Cada una con su README.md.
ROADMAP.md         Papers candidatos organizados por asignatura del máster; marcar los completados.
```

## Plantilla de review (`reviews/<slug>.md`)

```markdown
# <Título del paper>
**Autores:** · **Año:** · **Venue:** · **Enlace/DOI:**
**Campo:** física / matemáticas / ML · **Leído:** <fecha>

## TL;DR (3-5 líneas)
## Contexto y motivación
## Metodología
   (con las ecuaciones clave en LaTeX: $...$)
## Resultados principales
## Puntos fuertes y limitaciones
## Ideas de implementación
   (qué merece la pena reproducir y con qué alcance)
## Conexiones
   (con otros papers ya revisados — enlazar sus reviews)
```

## Convenciones de implementación

- **Python ≥ 3.11.** Stack base: `numpy`, `scipy`, `matplotlib`; `pandas`/`polars` para datos; `torch` para deep learning; `sympy` para cálculo simbólico.
- Cada implementación es autocontenida en `implementations/<slug>/` con:
  - `README.md`: qué parte del paper reproduce, cómo ejecutarlo, y comparación de resultados con los del paper.
  - `DERIVATIONS.md`: **el desarrollo matemático completo**, en inglés (ver abajo).
  - Scripts `.py` ejecutables; notebooks solo para exploración y figuras finales.
  - `requirements.txt` si necesita algo fuera del stack base.
- Priorizar claridad pedagógica sobre rendimiento: el objetivo es entender el paper, no producción.
- Referenciar en el código la ecuación o sección del paper que implementa (p. ej. `# Eq. (12), Sec. 3.2`).
- Validar siempre contra resultados del paper (una figura, una tabla o un valor numérico) y dejar constancia en el README.

## `DERIVATIONS.md` — el desarrollo matemático

Un `.md` por implementación, **en inglés**, con la matemática del tema desarrollada **de principio a fin**, para leerse seguido como un capítulo, no para consultarse.

- **Orden lógico y progresivo, no el del paper.** El documento es un desarrollo propio: se parte del problema, se simplifica, se resuelve en los casos fáciles y se va generalizando. Cada sección abre diciendo qué hace falta a continuación y por qué, y se apoya en lo ya deducido. **No es una auditoría del paper fórmula a fórmula.**
- **Conectar, no enumerar.** Cuando una misma idea reaparece (una no-derivabilidad, un autovector, un truco de linealización), decirlo explícitamente: el valor está en la red, no en las piezas sueltas.
- La numeración del paper se cita como **referencia al pasar** ("esto es su Eq. 3"), al final de la deducción y no como titular. Nada de abrir cada sección citando lo que el paper afirma.
- Toda la matemática en LaTeX, en bloque `$$...$$` cuando es un paso central. Sin saltarse pasos "obvios".
- Registro matemático estándar en inglés y en primera persona del plural (*"we now show that"*, *"it follows that"*, *"note that"*). El documento vale como muestra de escritura técnica, así que la prosa importa tanto como las cuentas.
- **Con figuras incrustadas**, generadas por un script propio (p. ej. `derivation_figures.py`) y **calculadas con el solver**, no dibujadas a mano — así cada figura comprueba lo que acompaña en vez de ilustrarlo. Debajo de cada una, un párrafo diciendo qué mirar.
- Índice al principio con enlaces internos.
- Si aparece una errata, un límite de validez que falta o una ambigüedad, va **en su sitio dentro del desarrollo**, con la comprobación numérica que lo respalda — no en una sección de erratas aparte.

## Flujo de trabajo

El comando `/paper` ejecuta el flujo completo (leer PDF → review → propuesta de implementación). Ver `.claude/skills/paper/SKILL.md`.
