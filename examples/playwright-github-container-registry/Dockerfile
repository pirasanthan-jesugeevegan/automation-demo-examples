# Base image
FROM node:14

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgtk-3-dev \
    libnotify-dev \
    libgconf-2-4 \
    libnss3 \
    libxss1 \
    libasound2 \
    xvfb \
    curl \
    gpg \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright
RUN npm install -g playwright

# Install browser dependencies for Chromium, Firefox, and WebKit
RUN playwright install-deps

# Set up a working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install application dependencies
RUN npm ci --only=production

# Copy the rest of the application code
COPY . .

# Set up the entry point command
CMD ["npm", "start"]
