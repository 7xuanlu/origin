<!-- README_SYNC: source=README.md sha256=7c7d6c8fba5ea5036d603ad2670082ab97e41a3629c9fee2b1d790fd98384738 -->

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/readme-banner-mobile.png">
    <img src="./docs/assets/readme-banner.png" alt="Wenlan: tu base de conocimientos respaldada por fuentes, diseñada para acumular valor con el tiempo." width="100%">
  </picture>
</p>

El trabajo útil con IA no debería desaparecer cuando termina una conversación. Wenlan construye las páginas adecuadas y las mantiene actualizadas a medida que las fuentes cambian, solicitando intervención solo cuando se requiere criterio humano.

<p align="center">
  <a href="./README.md">English</a> | <a href="./README.zh-Hans.md">简体中文</a> | <a href="./README.zh-Hant.md">繁體中文</a> | Español
</p>

<p align="center">
  <a href="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml?query=branch%3Amain"><img alt="CI" src="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://github.com/7xuanlu/wenlan/releases/latest"><img alt="Última versión" src="https://img.shields.io/github/v/release/7xuanlu/wenlan?sort=semver&label=release"></a>
  <a href="#license"><img alt="Licencia: Apache-2.0 y AGPL-3.0" src="https://img.shields.io/badge/license-Apache--2.0%20%2B%20AGPL--3.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#start-in-30-seconds">Primeros&nbsp;pasos</a> ·
  <a href="#what-does-wenlan-build">¿Qué&nbsp;es&nbsp;esto?</a> ·
  <a href="#what-can-it-do">Capacidades</a> ·
  <a href="#how-does-it-work">Flujo&nbsp;diario</a> ·
  <a href="#evaluation">Evaluación</a> ·
  <a href="#learn-more">Leer&nbsp;más</a>
</p>

https://github.com/user-attachments/assets/77272089-84ea-4eb0-a074-c9cc8b7b28fd

<p align="center">
  <sub>Una Página mantenida en la aplicación de escritorio: abre cualquier cita para inspeccionar la Fuente o Memoria detrás de la afirmación.</sub>
</p>

---

<a id="quickstart"></a>
<a id="start-in-30-seconds"></a>

## Primeros pasos

Wenlan funciona como un único daemon local. La aplicación de escritorio lo lleva dentro; la instalación sin interfaz te da ese mismo daemon sin ventana. En ambos casos, tus clientes de IA acceden a la misma base de conocimiento.

<a id="start-with-the-app"></a>
<a id="open-the-wiki"></a>
<a id="desktop-app"></a>

### Aplicación de escritorio

Descarga desde la [página de Releases](https://github.com/7xuanlu/wenlan/releases/latest):

- **macOS (Apple Silicon):** abre el `.dmg` y arrastra Wenlan a Aplicaciones. La aplicación está firmada y notarizada, así que no hay ningún aviso en el primer arranque. Desde el terminal en su lugar: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/scripts/install-macos-app.sh)"` (descarga, comprueba el SHA-256, la mueve a Aplicaciones).
- **Windows x64:** ejecuta el `-setup.exe`. Todavía no está firmado, así que cuando SmartScreen muestre "Windows protegió su PC", elige "Más información" y luego "Ejecutar de todas formas".
- **Linux:** todavía no tiene versión de escritorio; usa el entorno de ejecución sin interfaz de abajo.

La aplicación incluye el daemon, la CLI y el conector MCP, arranca el daemon al abrirse y ofrece conectar los clientes de IA que detecta: el plugin para Claude Code y Codex, una entrada MCP para el resto. Para actualizar, arrastra la aplicación nueva sobre la antigua y ábrela (Wenlan 0.17.0 y anteriores hay que cerrarlos a mano antes).

<a id="claude-code-in-30-seconds"></a>

<a id="codex-plugin"></a>

<a id="mcp-setup"></a>
<a id="mcp-clients"></a>

### Configuración con tu IA

Pega esto en Claude Code, Codex o cualquier otra herramienta que pueda seguir una guía de configuración:

```text
Configura Wenlan para este cliente de IA siguiendo:
https://raw.githubusercontent.com/7xuanlu/wenlan/main/docs/setup-with-ai.md

Instala solo lo que este cliente necesite. Luego verifica el entorno de ejecución local,
su conexión con Wenlan y un ciclo completo de captura/recuerdo.
```

La guía detecta qué cliente estás utilizando y mantiene los comandos específicos del cliente fuera de este README. No configura todas las herramientas de IA a menos que se lo pidas.

¿Necesitas solo el entorno de ejecución (headless) en macOS Apple Silicon?

```bash
npx -y wenlan setup
```

Esto descarga la CLI precompilada, el daemon y el conector MCP, inicia el entorno local y lo verifica. No se requiere toolchain de Rust ni Cargo. Linux x64/ARM64 con glibc tiene una [ruta de configuración automática de shell](docs/setup-with-ai.md#install-the-runtime); Windows x64 utiliza el archivo correspondiente de [Releases](https://github.com/7xuanlu/wenlan/releases/latest). macOS Intel actualmente [no tiene una instalación completa soportada del runtime](crates/wenlan-cli/README.md#macos-intel).

Instrucciones manuales y específicas por cliente: [Configuración asistida por IA](docs/setup-with-ai.md) · [Plugin de Claude Code](plugin/.claude-plugin/README.md) · [Plugin de Codex](plugin-codex/README.md) · [CLI y MCP](crates/wenlan-cli/README.md).

---

<a id="what-does-wenlan-build"></a>
<a id="why-it-compounds"></a>

## ¿Qué es esto?

Wenlan convierte documentos, notas y conversaciones pasadas con IA en una base de conocimientos respaldada por fuentes que se mantiene actualizada a medida que tu trabajo evoluciona. Las fuentes siguen siendo rastreables; las decisiones, lecciones y correcciones se convierten en memorias duraderas; ambas pueden sustentar las mismas Páginas mantenidas.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-system-mobile.png">
    <img src="./docs/assets/wenlan-system.png" alt="Las fuentes y memorias sustentan independientemente una Página mantenida. Wenlan puede reconstruir una Página obsoleta a partir de su sustento actual; la revisión de conflictos opcional puede resaltar conflictos protegidos, y los cambios en la escritura humana esperan al usuario." width="100%">
  </picture>
</p>

<a id="what-wenlan-is-not"></a>

**Construido para trabajos continuos.** Wenlan es para investigadores, escritores, consultores, equipos de producto y equipos de software cuyo conocimiento está disperso en documentos, notas y conversaciones de IA. Convierte ese material en Páginas inspeccionables que pueden mejorar a través de proyectos y semanas, no en otro historial de chat o almacén de memoria aislado. No es un sistema de gestión de vida ni un SDK de memoria embebido dentro de otro producto.

**Un sistema de conocimiento, tres roles:**

- **Las Fuentes mantienen rastreable el material que lee Wenlan.** Las conversaciones importadas permanecen como registros capturados; los archivos registrados sincronizan su contenido actual a medida que cambian.
- **Las Memorias preservan lo que el trabajo te enseña.** Los agentes capturan decisiones atómicas, lecciones, correcciones y sustituciones con procedencia.
- **Las Páginas compilan el conocimiento actual.** Wenlan convierte Fuentes y Memorias relevantes en Markdown con citas de fuente que puedes reutilizar, actualizar y revisar.

**La base de LLM-wiki, extendida:**

- **[LLM-wiki v1](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):** Karpathy definió Fuentes inmutables, una Wiki en Markdown mantenida por IA y un Esquema de reglas co-evolutivas para estructurarla y mantenerla. Wenlan implementa esa base con [campos de Memoria tipados](docs/technical-foundations.md#typed-memory-schema) y reglas integradas para la estructura de Páginas, procedencia, citas, actualización, propiedad y revisión.
- **[LLM-wiki v2](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2):** Rohitg00 añadió un ciclo de vida de memoria. Wenlan concreta esa dirección con Fuentes rastreables, Memorias atómicas estilo Zettelkasten capturadas por agentes (una idea completa por cada una) y Páginas mantenidas construidas a partir de ambas.

**El elemento distintivo de Wenlan:** Las Fuentes y las Memorias atómicas sustentan independientemente las Páginas mantenidas. El historial de Memoria preserva cómo cambió el conocimiento; el historial de la Página muestra qué evidencia actual sustenta la síntesis. Las Páginas mantenidas por máquina pueden reconstruirse a partir del sustento actual, mientras que los cambios en la escritura humana esperan como revisiones revisables.

<a id="knowledge-graph"></a>

### Un grafo de conocimiento que se vuelve más útil con el tiempo

El grafo de entidad-relación es una parte de la wiki conectada más amplia de Wenlan. Las **Páginas de Conocimiento** contienen la síntesis mantenida, las **Entidades** anclan personas, proyectos y conceptos reutilizables, las **Páginas de Fuente** hacen que el material importado o sincronizado sea inspeccionable, y las **Memorias** atómicas preservan decisiones y cambios. Funcionan a través de enlaces separados y explícitos: wikilinks de Página a Página, evidencia de Página, enlaces de Memoria a Entidad y relaciones de Entidad dirigidas.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-knowledge-network-mobile.png">
    <img src="./docs/assets/wenlan-knowledge-network.png" alt="Modelo conceptual del sistema de conocimiento conectado de Wenlan, con Páginas de Conocimiento, Páginas de Fuente, Memorias atómicas y Entidades conectadas a través de enlaces de Página, evidencia, enlaces de Memoria a Entidad y relaciones de Entidad." width="100%">
  </picture>
</p>

Dentro del grafo de entidades, un modelo de enriquecimiento configurado extrae Entidades tipadas, observaciones y relaciones dirigidas a partir de las Memorias. El enlace y la resolución de entidades reutilizan nodos existentes en lugar de tratar cada mención como nueva; cada Memoria conserva su Fuente y puede vincularse a múltiples Entidades. [Cómo se almacena el modelo conectado ->](docs/technical-foundations.md#connected-knowledge-model)

- **Significado y dirección:** Las relaciones utilizan un vocabulario predefinido como `uses` (usa), `part_of` (parte de), `contradicts` (contradice) y `replaced_by` (reemplazado por); los tipos desconocidos se reasignan a `related_to` (relacionado con) y se convierten en propuestas de vocabulario revisables.
- **Fuerza y procedencia:** Una relación puede almacenar confianza, una explicación y su Memoria de origen, para que las afirmaciones más fuertes y más débiles sigan siendo distinguibles e inspeccionables.
- **Comunidades que se enriquecen con el tiempo:** La propagación de etiquetas agrupa Entidades por densidad de relación, ponderada por el recuento de relaciones entre cada par. Estos grupos pueden organizar resúmenes de corpus opcionales mientras que los enlaces de Entidad añaden contexto de recuperación.
- **Corrección sin borrado:** Las afirmaciones relacionadas, las correcciones y las sustituciones explícitas permanecen inspeccionables juntas mientras se conservan las Fuentes originales y el historial de Memoria.

Durante la recuperación, la coincidencia densa de entidades encuentra entidades relevantes para la consulta. Cuando existen enlaces de grafo elegibles, el flujo de grafo-memoria predeterminado potencia las Memorias vinculadas como una tercera señal de [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf). La ruta depende de los datos y el alcance, y los límites de Espacio (Space) se siguen aplicando. [Cómo funciona la ruta del grafo ->](docs/technical-foundations.md#graph-assisted-retrieval)

<a id="retrieval"></a>

### Recuperación a través de palabras, significado y conexiones

La búsqueda central de Wenlan es un pipeline híbrido local, no una simple búsqueda de vectores. Cada etapa tiene una tarea diferente:

- **Coincidencia literal, [SQLite FTS5](https://www.sqlite.org/fts5.html):** un índice de texto completo encuentra términos literales, identificadores y frases.
- **Significado similar, FastEmbed + [`Qdrant/bge-base-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q):** un modelo inglés cuantizado crea embeddings de 768 dimensiones; [libSQL cosine DiskANN](https://turso.tech/blog/approximate-nearest-neighbor-search-with-diskann-in-libsql) los indexa para la recuperación de vecinos más cercanos aproximados.
- **Clasificación combinada, [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) ponderado (`k = 60`):** las listas de clasificación léxica y semántica se fusionan sin fingir que sus puntuaciones brutas comparten una escala; la similitud de coseno también pondera la contribución del vector.
- **Contexto conectado, flujo de grafo-memoria:** los enlaces de entidad elegibles añaden una tercera señal RRF mientras que el alcance de lectura activo sigue filtrando las Memorias devueltas.
- **Precisión opcional, re-clasificación por cross-encoder:** a diferencia de los embeddings, [`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en) o [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) lee cada par consulta-candidato y reordena el grupo más pequeño; la re-clasificación está desactivada por defecto.

Los canales de Página, episódicos y de hechos son opcionales y recurren a las señales de búsqueda restantes si no están disponibles. El Espacio sigue limitando el alcance de lectura. [Métodos, valores predeterminados y limitaciones ->](docs/technical-foundations.md)

<a id="what-makes-wenlan-distinct"></a>
<a id="why-is-wenlan-different"></a>
<a id="two-lifecycles"></a>

### Dos ciclos de vida, un sistema de conocimiento mantenido

Una wiki generada puede quedar obsoleta; un almacén de memoria puede fragmentarse en hechos desconectados. Wenlan vincula dos ciclos de vida sin colapsarlos en una sola capa.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-lifecycle-mobile.png">
    <img src="./docs/assets/wenlan-lifecycle.png" alt="Una memoria anterior permanece vinculada después de una captura de sustitución explícita. Cuando una Página está obsoleta, Wenlan la reconstruye a partir de Fuentes y Memorias actuales, registra la revisión y pone los cambios en la escritura humana en espera para revisión." width="100%">
  </picture>
</p>

#### Memoria Atómica

`CAPTURA -> CLASIFICA -> ENRIQUECE -> VINCULA -> RECONCILIA`

La captura y la sustitución explícita son fundamentales. Las etapas basadas en modelos se ejecutan solo cuando el modelo correspondiente está configurado, y el paso de reconciliación está desactivado por defecto.

| Operación | Lo que hace Wenlan |
|---|---|
| **Captura** | Los agentes escriben una idea completa y autónoma por Memoria, siguiendo el principio de nota atómica de Zettelkasten en lugar de guardar toda la conversación. |
| **Clasifica** | Con el modelo en el dispositivo, Wenlan asigna `identity` (identidad), `preference` (preferencia), `decision` (decisión), `lesson` (lección), `gotcha` (advertencia) o `fact` (hecho); un tipo preciso proporcionado por el cliente tiene prioridad. |
| **Enriquece** | Con el modelo en el dispositivo, añade campos estructurados, pistas de recuperación, fechas de eventos, calidad, importancia y etiquetas cuando estén disponibles. |
| **Vincula** | Mantiene la procedencia y, cuando el enriquecimiento está habilitado, conecta Memorias con entidades y relaciones en el grafo de conocimiento. |
| **Reconcilia** | Los reemplazos explícitos preservan una cadena de `supersedes` (sustituye). Un reemplazo de un agente cuyo nivel de confianza sea inferior a "full" se pone en cola para revisión humana automáticamente, sin necesidad de ninguna opción. Un paso opcional en el dispositivo también puede poner en cola conflictos protegidos para revisión en lugar de sobrescribir el historial; ese paso está desactivado por defecto y debe habilitarse explícitamente. |

Configuración avanzada: establece `WENLAN_ENABLE_DUAL_POOL_RESOLVE=1` para habilitar ese paso de reconciliación.

#### Página Mantenida

`DESTILA -> CITA -> RASTREA -> ACTUALIZA -> REVISA`

| Operación | Lo que hace Wenlan |
|---|---|
| **Destila** | Compila Fuentes y Memorias relacionadas en una Página Markdown. |
| **Cita** | Mantiene los registros de citas y el estado de verificación; la actualización automática descarta un borrador cuando falla la verificación del respaldo de las citas. |
| **Rastrea** | Registra qué evidencia sustenta la Página, por qué quedó obsoleta y un registro de cambios limitado. |
| **Actualiza** | Cuando una Página se marca como obsoleta, reconstruye las Páginas mantenidas automáticamente que cumplen los requisitos a partir de la evidencia actual. |
| **Revisa** | Convierte los cambios en una Página que editaste en una revisión propuesta en lugar de una reescritura silenciosa. |

Por ejemplo, importa un documento de diseño y captura una decisión de depuración en Codex. Wenlan puede compilar una Página que cite ambos. Cuando esa Página se actualice, se reconstruirá a partir de su sustento actual; si la has editado, el cambio propuesto esperará revisión.

<a id="local-markdown"></a>

### Markdown local que funciona con Obsidian

Tu síntesis duradera permanece en archivos ordinarios en lugar de un formato de editor propietario:

- **Archivos planos:** Las Páginas y notas de sesión permanecen como Markdown en `~/.wenlan/`.
- **Historial inspeccionable:** Los flujos de destilación y entrega pueden registrar lotes lógicos de archivos mediante commits en un repositorio git local.
- **Coexistencia con Obsidian:** Wenlan lee un vault existente como una fuente. Crea un enlace simbólico de `~/.wenlan/pages/` hacia el vault o exporta una Página desde la aplicación de escritorio; tus ediciones siguen siendo propiedad humana, y las actualizaciones posteriores de la máquina se convierten en revisiones revisables.

El historial local es directamente inspeccionable:

```text
$ git -C ~/.wenlan log --oneline
a1b2c3d distill: 4 pages
9f8e7d6 session: embedding-work
```

---

<a id="what-you-get"></a>
<a id="what-can-it-do"></a>
<a id="what-can-i-bring-in"></a>

## Capacidades

- **Importación de chats:** Importa archivos ZIP de exportación de ChatGPT o Claude; Wenlan omite automáticamente las conversaciones ya importadas.
- **Fuentes de documentos:** Ingiere un archivo `.md`, `.txt` o `.pdf` del que se pueda extraer texto; procesa de forma recursiva una carpeta que los contenga; o indexa el Markdown de un vault de Obsidian.
- **Sincronización incremental:** Las Fuentes de archivos y carpetas regulares rastrean los cambios en segundo plano; los vaults de Obsidian permanecen de solo lectura y se resincronizan bajo demanda.
- **Memoria Atómica:** Los clientes MCP guardan una sola decisión, lección, corrección, preferencia o hecho completo, con [procedencia y sustitución](https://wenlan.app/learn/ai-memory-provenance) que registran de dónde vino y qué reemplaza.
- **[Enriquecimiento tipado](docs/technical-foundations.md#typed-memory-schema):** Un modelo configurado clasifica cada Memoria y luego añade los campos estructurados definidos para su tipo, además de fechas, etiquetas, pistas de recuperación y enlaces de grafo.
- **[Páginas respaldadas por fuentes](https://wenlan.app/docs/source-backed-pages):** Destila Fuentes y Memorias relacionadas en Páginas Markdown con referencias de fuente y `[[wikilinks]]`; el daemon puede verificar y registrar citas por afirmación.
- **Actualización condicionada por citas:** La actualización automática rechaza borradores con pocas citas; las Páginas de máquina se actualizan mientras que las ediciones humanas se convierten en revisiones revisables.
- **[Recuperación híbrida](docs/technical-foundations.md#retrieval-pipeline):** FTS5 encuentra palabras exactas, embeddings BGE locales encuentran el significado y RRF fusiona sus rangos; los enlaces de grafo pueden añadir contexto.
- **[Canales de recuperación](docs/technical-foundations.md#optional-channels-and-defaults):** Canales opcionales de Página, episódicos y por hecho amplían la recuperación; la re-clasificación por cross-encoder puede mejorar la precisión.
- **[Grafo de conocimiento](docs/technical-foundations.md#graph-data-and-entity-resolution):** Entidades tipadas, relaciones y observaciones conectan personas, proyectos, afirmaciones y Memorias de apoyo.
- **[Revisión con intervención humana](https://wenlan.app/docs/review-and-trust):** El trabajo rutinario sigue siendo automático; los conflictos protegidos, las revisiones de Páginas, las fusiones de entidades y el vocabulario nuevo esperan juicio humano.
- **[Espacios](https://wenlan.app/docs/spaces):** Mantén el conocimiento laboral, personal, de clientes y de repositorio dentro de un alcance de recuperación explícito.
- **[Daemon local + MCP](https://wenlan.app/docs/architecture):** Un daemon de Rust ligero es la única fuente de verdad local. La aplicación de escritorio y la CLI lo llaman directamente; los clientes de IA utilizan pequeños conectores MCP para acceder al mismo conocimiento.
- **Integraciones personalizadas:** La API HTTP de localhost acepta texto preparado, contenido de páginas web y Memorias de otros flujos de captura.
- **Mantenimiento en segundo plano:** El daemon sigue trabajando después de cerrar la aplicación de escritorio, ejecutando la sincronización configurada, el enriquecimiento, el trabajo de citas y la actualización de Páginas elegibles.
- **[Elección de modelo](docs/technical-foundations.md#model-roles):** La recuperación base permanece local; el enriquecimiento y la síntesis pueden usar Qwen en el dispositivo, un endpoint local o un modelo en la nube configurado.
- **[Propiedad inspeccionable](https://wenlan.app/learn/markdown-local-index-ai-memory):** Las Memorias y los datos del grafo permanecen en libSQL local; el Markdown, las citas, las revisiones, el historial de git y las exportaciones de Obsidian permanecen inspeccionables.
- **Comprobaciones de estado de solo lectura:** [`doctor`](https://wenlan.app/docs/diagnostics-and-issue-reports) verifica el runtime; [`lint`](plugin/skills/lint/SKILL.md) encuentra citas mal formadas, enlaces huérfanos, embeddings rotos y problemas de integridad del índice de búsqueda o del grafo sin reescribir el conocimiento.

---

<a id="how-wenlan-works"></a>
<a id="how-does-it-work"></a>

## Flujo diario

El sistema anterior se convierte en un pequeño bucle diario: comienza con el conocimiento relevante, captura lo que importa mientras trabajas, cierra con una entrega (handoff) y deja que Wenlan refine lo que debería volver la próxima vez. Cada paso deja la misma base de conocimientos más refinada en lugar de crear otro historial desconectado.

El bucle tiene cuatro pasos:

1. **Encontrar el conocimiento actual.** Abre una Página relevante, busca o usa `/recall <consulta>`; `/brief [tema]` lee el Brief del Espacio actual y, si proporcionas un tema, añade contexto etiquetado por separado de ese mismo Espacio. Los clientes sin comandos de plugin usan las herramientas equivalentes de página, búsqueda, recuerdo y brief.
2. **Capturar y encontrar conocimiento mientras trabajas.** `/capture <cosa>` guarda una decisión, lección, advertencia o hecho con su fuente. `/recall <consulta>` recupera solo lo que es relevante en lugar de cargar todo tu historial.
3. **Cerrar el bucle.** `/handoff` registra qué cambió y aplica actualizaciones tipadas a cada elemento del Brief del Espacio actual.
4. **Mantener la wiki actualizada.** `/distill` crea o actualiza páginas deliberadamente. Entre sesiones, pasos opcionales basados en modelos pueden enriquecer capturas, conectar entidades relacionadas y actualizar páginas elegibles. `/lint` verifica la salud del conocimiento; `/curate` te presenta las revisiones propuestas y cualquier elemento de revisión de conflictos creado por el paso de reconciliación opcional.

### Cola offline (outbox)

Si el daemon local no está accesible, `wenlan capture` y `wenlan brief update` escriben sus solicitudes en una cola local duradera (outbox) y terminan correctamente. Cuando el daemon vuelve, drena esas escrituras por las rutas HTTP normales; revisa la cola con `wenlan outbox status` o pide una reproducción inmediata con `wenlan outbox drain`. Una escritura que el daemon rechaza de plano (un 4xx, por ejemplo al no pasar el control de calidad del contenido) se mueve a `outbox/failed/` con un recibo en lugar de reintentarse para siempre; un fallo de transporte o un error del servidor (5xx) la deja en la cola para el siguiente drenaje, que se ejecuta automáticamente cada 60 segundos.

### Modelos y privacidad

- **Recuperación base local:** El [modelo de embedding BGE](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q) se ejecuta a través de FastEmbed en tu máquina para la búsqueda híbrida y no necesita clave de API.
- **Síntesis opcional en el dispositivo:** El enriquecimiento y la síntesis de Páginas pueden usar [`Qwen3 4B`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) o [`Qwen3.5 9B`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), según la elección del usuario, a través de [llama.cpp](https://github.com/ggml-org/llama.cpp). Wenlan no descarga ni activa un modelo de lenguaje hasta que elijas uno.
- **Otros proveedores:** Un endpoint local compatible con OpenAI como Ollama o LM Studio, o un proveedor en la nube configurado, pueden suministrar el enriquecimiento y la síntesis basados en modelos.
- **Divulgación de nube:** Si el endpoint del modelo que seleccionas es remoto, Wenlan envía los prompts del sistema y del usuario de esa tarea a ese endpoint. La recuperación local y la síntesis en el dispositivo permanecen en tu máquina.
- **Sin telemetría:** Wenlan no envía telemetría.

Referencia completa del flujo de trabajo: [plugin/skills](plugin/skills/README.md). Roles técnicos del modelo: [fundamentos técnicos](docs/technical-foundations.md#model-roles).

### Tus datos y la desinstalación

Nada queda encerrado. Las páginas y las notas de sesión son Markdown en `~/.wenlan/`; las memorias viven en una base de datos libSQL dentro del directorio de datos de la plataforma (`~/Library/Application Support/wenlan/` en macOS, `~/.local/share/wenlan/` en Linux, `%LOCALAPPDATA%\wenlan\` en Windows). Copia esas dos carpetas para hacer una copia de seguridad o mover tu Wenlan. Si esta instalación se actualizó desde Origin, todavía conserva una copia completa de sus datos en `~/.origin/` y en la carpeta hermana de datos `origin` (`~/Library/Application Support/origin/` en macOS, `~/.local/share/origin/` en Linux, `%LOCALAPPDATA%\origin\` en Windows); borra o copia también esas dos.

Para desinstalar: el interruptor *Ejecutar Wenlan en segundo plano al iniciar sesión* de la app elimina el registro de arranque — desactívalo, cierra la app y borra `Wenlan.app` o ejecuta el desinstalador de Windows, y después borra las carpetas anteriores. `wenlan background off` solo detiene el daemon y desactiva el arranque automático; no elimina el registro de arranque, así que una instalación solo de CLI debe seguir en su lugar el punto de desinstalación del daemon en [PRIVACY.md](docs/PRIVACY.md). Las rutas que Wenlan escribe están ahí.

---

<a id="evaluation"></a>

## Evaluación

Esto es una instantánea de solo recuperación, no una afirmación sobre la calidad de la respuesta de extremo a extremo. El método, los recibos del entorno y el flujo de actualización residen en [docs/eval](docs/eval/README.md).

<!-- EVAL_SNAPSHOT_START -->
| Benchmark | Recall@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| LME_Oracle (500 Q) | 93.6% | 0.857 | 0.883 |
| LME_S (deep, 90 Q) | 87.7% | 0.815 | 0.822 |
<!-- EVAL_SNAPSHOT_END -->

---

<a id="learn-more"></a>

## Leer más

Documentación más detallada, conceptos y comparaciones:

### Documentación

- [Primeros pasos](https://wenlan.app/docs/get-started): instala y verifica el primer bucle local.
- [Flujo diario](https://wenlan.app/docs/daily-workflow): brief, capture, recall, handoff, distill, lint y curate.
- [Clientes MCP](https://wenlan.app/docs/mcp-clients): conecta Claude Code, Codex, Cursor, Claude Desktop y otros clientes.

### Guías de flujo de trabajo

- [Crear una base de conocimiento de proyectos para consultoría](https://wenlan.app/learn/build-client-project-knowledge-base-for-consulting)
- [Crear una base de conocimiento para investigación de inversiones](https://wenlan.app/learn/build-investment-research-knowledge-base)
- [Crear una base de conocimiento de investigación de producto antes de redactar un PRD](https://wenlan.app/learn/build-product-research-knowledge-base-for-prd)
- [Crear una base de conocimiento de incidentes SRE](https://wenlan.app/learn/build-sre-incident-knowledge-base)
- [Crear una base de conocimiento de definiciones de métricas de negocio](https://wenlan.app/learn/build-business-metric-definition-knowledge-base): convierte especificaciones de KPI aprobadas en un diccionario de datos respaldado por fuentes, con texto de fórmula, granularidad, exclusiones, propietarios, revisiones y estado de revisión.

### Conceptos

- [Por qué una wiki viva, no solo memoria de IA](https://wenlan.app/learn/ai-work-memory): el problema y el modelo de producto en profundidad.
- [Servidor de memoria MCP](https://wenlan.app/learn/mcp-memory-server): cómo Wenlan expone el conocimiento a través de herramientas de IA.
- [Memoria de IA local-first](https://wenlan.app/learn/local-first-ai-memory): datos, privacidad y control.
- [Markdown e índice local](https://wenlan.app/learn/markdown-local-index-ai-memory): almacenamiento, recuperación y propiedad.
- [Bucle de entrega de agentes de IA](https://wenlan.app/learn/ai-agent-handoff-loop): cómo trasladar el trabajo limpiamente a la siguiente sesión.
- [Base de conocimiento para investigación](https://wenlan.app/learn/source-backed-research-knowledge-base): convierte artículos seleccionados en una matriz bibliográfica y una síntesis verificable con fuentes.

### Comparaciones

- [Wenlan vs Basic Memory](https://wenlan.app/learn/wenlan-vs-basic-memory)
- [Wenlan vs claude-mem](https://wenlan.app/learn/wenlan-vs-claude-mem)
- [Wenlan vs Superlocal Memory](https://wenlan.app/learn/wenlan-vs-superlocal-memory)

---

## Contribuir

Las correcciones de errores, casos de evaluación, documentación y funciones son bienvenidos. Instalar Wenlan no requiere compilar desde el código fuente. Para el desarrollo local, ejecuta estos comandos desde la raíz de este repositorio:

```bash
# crates del daemon (default-members: la aplicación de escritorio no se compila)
cargo build
cargo test

# aplicación de escritorio (target de Cargo y herramientas frontend en la raíz)
pnpm install
pnpm dev:all
pnpm build:all
```

`pnpm dev:all` es el punto de entrada de desarrollo admitido para la aplicación de escritorio. Mantiene los puertos de desarrollo, los datos, la propiedad de los procesos, la identidad de la aplicación, los sockets MCP y el estado de Remote Access separados del runtime de producción instalado; una compilación de depuración iniciada sin ese aislamiento se niega a ejecutarse. Consulta el [AGENTS.md](AGENTS.md) y [CONTRIBUTING.md](.github/CONTRIBUTING.md) de este repositorio, además del [app/AGENTS.md](app/AGENTS.md) dentro del repositorio, para el flujo de trabajo de desarrollo completo. Reportes de seguridad: [SECURITY.md](.github/SECURITY.md). Política de privacidad: [PRIVACY.md](docs/PRIVACY.md). Por favor, lee también el [Código de Conducta](.github/CODE_OF_CONDUCT.md).

---

<a id="code-signing-policy"></a>

## Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).

- **Autores:** [@7xuanlu](https://github.com/7xuanlu), que puede hacer commit en este repositorio sin una revisión adicional.
- **Revisores:** [@7xuanlu](https://github.com/7xuanlu). Todo cambio de alguien que no sea committer llega como pull request y se revisa antes de fusionarse.
- **Aprobadores:** [@7xuanlu](https://github.com/7xuanlu), que aprueba cada solicitud de firma y así decide qué versión se firma.

La autenticación multifactor es obligatoria para cada mantenedor, en GitHub y en SignPath, y nadie se añade a ninguno de los dos sin ella. Las versiones se compilan únicamente con el workflow de release por etiqueta de este repositorio, en runners alojados por GitHub, desde el commit al que apunta la etiqueta.

**Política de privacidad:** [PRIVACY.md](docs/PRIVACY.md) — qué guarda Wenlan, dónde lo guarda y cada caso que conocemos en que accede a la red. Cómo se firma cada plataforma: [docs/code-signing.md](docs/code-signing.md).

La solicitud a SignPath está pendiente. Los instaladores de Windows aún no están firmados.

---

<a id="license"></a>

## Licencia

Wenlan usa dos licencias, una por cada parte del repositorio.

- **Apache-2.0** ([`LICENSE`](LICENSE)) cubre el runtime local, la CLI, el servidor MCP, los tipos compartidos y los archivos del plugin de Claude Code y Codex. Constrúyelo libremente sobre esto.
- **AGPL-3.0-only** ([`app/LICENSE`](app/LICENSE)) cubre la aplicación de escritorio: el crate `app/` y el frontend de React que incluye. Si ejecutas una versión modificada de la aplicación como servicio en red, la AGPL te pide ofrecer ese código modificado a sus usuarios.

La separación es deliberada. El código Apache-2.0 puede usarse dentro de un programa AGPL-3.0, así que la aplicación de escritorio se apoya en el runtime sin que ninguna de las dos licencias se incumpla.

---

<a id="acknowledgments"></a>

## Linaje y pares

Wenlan (文瀾) toma su nombre de Wenlan Ge (文瀾閣), una biblioteca imperial que albergaba la Siku Quanshu como parte de una de las colecciones de libros más grandes de China.

El modelo llm-wiki v2 de Wenlan es su propia dirección de producto, informada por los linajes de LLM-wiki y memoria de agentes:

- La [nota de LLM-wiki de Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) estableció el patrón de fuente bruta a wiki mantenida.
- La [propuesta de LLM Wiki v2 de Rohitg00](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2) extiende ese patrón con ciclo de vida de memoria, confianza, grafo y mecanismos de recuperación. [agentmemory](https://github.com/rohitg00/agentmemory) es su implementación concreta de memoria de agente.
- [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) es una implementación completa de escritorio del patrón LLM-wiki centrado en documentos.
- [basic-memory](https://github.com/basicmachines-co/basic-memory), [obsidian-mind](https://github.com/breferrari/obsidian-mind), [mcp-memory-service](https://pypi.org/project/mcp-memory-service/), [Memoria](https://github.com/matrixorigin/Memoria) y [OpenMemory](https://github.com/CaviraOSS/OpenMemory) exploran formas adyacentes de conocimiento local y memoria de agentes.
