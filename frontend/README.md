# BlindSight Expo client

This is the primary BlindSight frontend. It runs on Android and iOS through Expo and exports a web
build that the repository's FastAPI app serves at `/`. The legacy browser client lives at
`/reference/`.

## Get started

1. Install dependencies.

   ```bash
   npm ci
   ```

2. Start a development server.

   ```bash
   npm start
   ```

In the output, you'll find options to open the app in a

- [development build](https://docs.expo.dev/develop/development-builds/introduction/)
- [Android emulator](https://docs.expo.dev/workflow/android-studio-emulator/)
- [iOS simulator](https://docs.expo.dev/workflow/ios-simulator/)
- [Expo Go](https://expo.dev/go), a limited sandbox for trying out app development with Expo

The routes live in `src/app`. API credentials can be supplied through the Settings screen or the
`EXPO_PUBLIC_BLINDSIGHT_API_URL` and `EXPO_PUBLIC_BLINDSIGHT_API_KEY` environment variables. Never
commit `.env.local`.

## Build the served web client

```bash
npm run build:web
```

This creates the ignored `dist/` directory consumed by `tools.local_dev` and `modal_app.py`.

### Other setup steps

- To set up ESLint for linting, run `npx expo lint`, or follow our guide on ["Using ESLint and Prettier"](https://docs.expo.dev/guides/using-eslint/)
- If you'd like to set up unit testing, follow our guide on ["Unit Testing with Jest"](https://docs.expo.dev/develop/unit-testing/)
- Learn more about the TypeScript setup in this template in our guide on ["Using TypeScript"](https://docs.expo.dev/guides/typescript/)

## Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.
