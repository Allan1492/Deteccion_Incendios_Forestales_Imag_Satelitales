# Libro de Préstamos

App web para gestionar clientes, préstamos y pagos. Corre en cualquier navegador,
se instala en pantalla de inicio en iPhone, y guarda todo en tu propia base de
datos (Firebase Firestore), separada por cuenta de usuario.

## 1. Requisitos

- [Node.js](https://nodejs.org) instalado (versión 18 o superior). Verifica con:
  ```bash
  node -v
  ```
- VS Code con la terminal integrada (Terminal → New Terminal).

## 2. Instalar dependencias

Abre esta carpeta en VS Code, abre la terminal, y corre:

```bash
npm install
```

## 3. Crear tu backend en Firebase

Sigue estos pasos una sola vez (10-15 min):

1. Ve a **https://console.firebase.google.com** → Crear proyecto.
2. **Authentication** → Comenzar → pestaña "Sign-in method" → habilita **Correo electrónico/contraseña**.
3. **Firestore Database** → Crear base de datos → modo producción → elige una región cercana.
4. En **Reglas** de Firestore, pega el contenido del archivo `firestore.rules` de este proyecto y publica.
5. Ícono de engranaje → Configuración del proyecto → "Tus apps" → ícono web `</>` → registra la app (sin marcar Hosting) → copia el objeto `firebaseConfig`.

## 4. Configurar tus variables de entorno

```bash
cp .env.example .env
```

Abre `.env` y pega cada valor de tu `firebaseConfig`:

```
VITE_FIREBASE_API_KEY=AIzaSy...
VITE_FIREBASE_AUTH_DOMAIN=tu-proyecto.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=tu-proyecto
VITE_FIREBASE_STORAGE_BUCKET=tu-proyecto.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123
```

**El archivo `.env` nunca se sube a git** (ya está en `.gitignore`). Si usas git/GitHub,
verifica siempre con `git status` que `.env` no aparezca en la lista antes de un commit.

## 5. Probarlo en local

```bash
npm run dev
```

Te da un link tipo `http://localhost:5173`. Ábrelo en tu navegador para probar. Para
probarlo desde tu iPhone en la misma red, usa la IP local que Vite muestra en la terminal
(algo como `http://192.168.x.x:5173`).

## 6. Publicarlo con su propio link (Firebase Hosting)

```bash
npm install -g firebase-tools
firebase login
firebase init hosting
```

Cuando pregunte:
- **"What do you want to use as your public directory?"** → responde `dist`
- **"Configure as a single-page app?"** → responde `Yes`
- **"Set up automatic builds with GitHub?"** → `No` (opcional, para más adelante)

Luego, cada vez que quieras publicar cambios:

```bash
npm run build
firebase deploy
```

Te da un link fijo tipo `https://tu-proyecto.web.app`. Ese es el que agregas a
pantalla de inicio en iPhone (Safari → compartir → "Agregar a pantalla de inicio").

## Sobre la seguridad de este proyecto

- **Las credenciales de Firebase (`firebaseConfig`) no son un secreto en el sentido
  clásico** — quedan visibles en cualquier app web de Firebase, incluso en producción,
  porque el navegador necesita saber a qué proyecto conectarse. Aun así, las sacamos a
  `.env` por buena práctica: mantiene el código limpio, evita subirlas por accidente a
  un repositorio público, y facilita tener credenciales distintas para desarrollo/producción
  más adelante.
- **La seguridad real vive en `firestore.rules`.** Esas reglas son las que de verdad
  impiden que alguien lea o escriba datos que no le pertenecen, y validan la forma de
  los datos (montos positivos, campos correctos, etc.) incluso si alguien intenta
  manipular las peticiones directamente. Cada vez que cambies la estructura de datos,
  actualiza también las reglas.
- **Autenticación por correo/contraseña** con verificación de correo al crear cuenta.
  Cada usuario solo ve sus propios datos (aislados por `uid` en la base de datos).
- **Antes de publicar en producción**, considera activar en la consola de Firebase:
  - **App Check** (Configuración del proyecto → App Check) — evita que alguien use tu
    base de datos desde fuera de tu app real.
  - Límites de intentos de inicio de sesión (Firebase ya aplica algunos por defecto).
- Si en algún momento compartes este repositorio (ej. en GitHub), verifica que sea
  **privado** o que `.env` de verdad no se haya subido nunca — revisa el historial de
  git si tienes dudas.

## Estructura del proyecto

```
libro-prestamos-app/
├── index.html          # Punto de entrada
├── src/
│   ├── main.js          # Toda la lógica: auth, base de datos, interfaz
│   └── style.css         # Estilos
├── firestore.rules      # Reglas de seguridad de la base de datos
├── .env.example         # Plantilla de variables de entorno
├── .env                 # Tus credenciales reales (no se sube a git)
└── package.json
```

## Próximos pasos pendientes del proyecto

- Recibos digitales descargables/compartibles
- Informe "pre-día de pago" para enviar al cliente
- Editar/eliminar clientes y préstamos
- Exportar historial completo (respaldo manual adicional)
