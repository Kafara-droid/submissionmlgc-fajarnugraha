FROM node:20

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

ENV PORT=3000

ENV MODEL=URL=https://storage.googleapis.com/model-cancer/model-in-prod/model.json

CMD ["npm", "start"]