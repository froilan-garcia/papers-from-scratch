# ClaudeLarp — Reviews de literatura + implementación de papers

Espacio de trabajo de Froilán (graduado en Física y Matemáticas, Univ. de Oviedo; máster en Big Data Analytics en la UC3M desde septiembre de 2026) para leer papers científicos, escribir reviews estructuradas e implementar su contenido en Python.

## Idioma

- Conversación y reviews: **español**.
- Código, nombres de variables y comentarios: **inglés** (convención estándar).
- Términos técnicos sin traducción asentada se dejan en inglés (p. ej. *embedding*, *gauge*).

## Estructura

```
papers/            PDFs originales. Nombre: <año>-<primer-autor>-<titulo-corto>.pdf
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
  - Scripts `.py` ejecutables; notebooks solo para exploración y figuras finales.
  - `requirements.txt` si necesita algo fuera del stack base.
- Priorizar claridad pedagógica sobre rendimiento: el objetivo es entender el paper, no producción.
- Referenciar en el código la ecuación o sección del paper que implementa (p. ej. `# Eq. (12), Sec. 3.2`).
- Validar siempre contra resultados del paper (una figura, una tabla o un valor numérico) y dejar constancia en el README.

## Flujo de trabajo

El comando `/paper` ejecuta el flujo completo (leer PDF → review → propuesta de implementación). Ver `.claude/skills/paper/SKILL.md`.
