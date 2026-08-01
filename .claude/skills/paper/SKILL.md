---
name: paper
description: Flujo completo de review e implementación de un paper científico en este repo. Usar siempre que el usuario invoque /paper, mencione que quiere leer/revisar/analizar un paper, pase la ruta o URL de un PDF académico, cite un título o identificador de arXiv/DOI, o pida "implementar" o "reproducir" el contenido de un artículo — aunque no diga explícitamente la palabra "review".
---

# Paper: review + implementación

Ejecuta el flujo completo del repo para un paper: conseguir el PDF → leerlo → escribir la review → implementar lo reproducible. Las convenciones de formato, plantilla de review y stack de Python están en el `CLAUDE.md` del repo; esta skill define el proceso, no el formato.

El argumento puede ser una ruta local a un PDF, una URL, o un título/identificador (arXiv, DOI). Si no hay argumento, pregunta qué paper quiere revisar.

## Paso 1 — Conseguir el PDF

- **Ruta local**: si no está ya en `papers/`, cópialo ahí renombrándolo a la convención `<año>-<primer-autor>-<titulo-corto>.pdf`. Ese slug (sin `.pdf`) se reutiliza en `reviews/` e `implementations/` — decídelo bien aquí.
- **URL**: descárgalo a `papers/` con ese mismo esquema de nombre. Para arXiv, usa la URL `https://arxiv.org/pdf/<id>`.
- **Título o referencia vaga**: busca en la web (arXiv, Semantic Scholar, Google Scholar) y confirma con el usuario que es el paper correcto antes de descargarlo, para no revisar el artículo equivocado.

## Paso 2 — Leer y hacer la review

Lee el PDF completo (usa la skill de PDF si hace falta OCR o extracción). Escribe `reviews/<slug>.md` en español siguiendo exactamente la plantilla del `CLAUDE.md`. Al escribirla:

- Transcribe las ecuaciones clave en LaTeX; son la base de la implementación posterior.
- En "Conexiones", revisa qué otras reviews existen ya en `reviews/` y enlaza las relacionadas — el valor del repo crece con esa red de conexiones.
- En "Ideas de implementación", sé concreto: qué figura, tabla o resultado numérico del paper se puede reproducir y con qué datos.

## Paso 3 — Proponer el alcance de la implementación

Antes de escribir código, resume al usuario qué propones implementar (qué resultado del paper se reproduce, qué se simplifica, tiempo estimado de cómputo) y espera su confirmación. Los papers a menudo requieren datos o cómputo inasequibles; acordar el alcance evita implementaciones inútiles.

## Paso 4 — El desarrollo matemático

Antes o en paralelo al código, escribe `implementations/<slug>/DERIVATIONS.md`: la matemática del tema desarrollada de principio a fin, en **orden lógico propio**, para leerse seguido. **Se escribe en inglés**, a diferencia de la review y del README. El formato está en el `CLAUDE.md`.

Lo primero es diseñar ese orden, y no sale del índice del paper. Pregúntate qué es lo mínimo para plantear el problema, qué lo simplifica, qué caso particular se resuelve entero a mano, y qué hace falta añadir para levantar cada restricción. Ese árbol de dependencias es el guion. Los papers exponen para convencer, no para construir: su orden casi nunca sirve.

Deduce **todo** lo que uses, incluido lo que el paper da por sabido — que suele ser mucho: los *"are easily shown to be"*, los *"one can show"*, las fórmulas enunciadas sin más. No des por bueno un paso porque "se ve": si al deducirlo aparece una errata, un límite de validez que falta o una ambigüedad, va documentado en su sitio con la comprobación numérica que lo respalde.

Las figuras se generan con un script propio y se calculan con el solver ya implementado, de modo que cada una valide lo que acompaña.

## Paso 5 — Implementar (poco a poco)

El usuario prefiere implementar de forma incremental: primero lee el paper, y luego se programa por partes a lo largo de varias sesiones — nunca todo el código de golpe. El objetivo es que él entienda cada pieza, así que implementa solo la parte que pida en cada momento, explica el código y conéctalo con las ecuaciones del paper.

Crea `implementations/<slug>/` siguiendo las convenciones del `CLAUDE.md` (scripts claros, referencias a ecuaciones del paper, `README.md` con instrucciones y comparación de resultados). Ejecuta cada parte y verifica que reproduce —al menos cualitativamente— el resultado del paper; deja constancia de la comparación en el README. Si los resultados no cuadran, dilo honestamente en el README en lugar de maquillarlo: una discrepancia documentada también es un resultado.

## Paso 6 — Cierre

Termina con un resumen: dónde quedó cada archivo, qué se reprodujo y qué quedó pendiente. Si el repo usa git, ofrece hacer commit.
