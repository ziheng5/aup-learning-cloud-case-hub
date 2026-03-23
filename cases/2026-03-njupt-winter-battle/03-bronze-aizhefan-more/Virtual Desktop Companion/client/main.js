const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 400,
        height: 600,
        x: 100,
        y: 100,
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        skipTaskbar: true,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    mainWindow.loadFile('index.html');
    
    mainWindow.setIgnoreMouseEvents(false);
}

app.whenReady().then(() => {
    createWindow();
    
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

ipcMain.handle('minimize-window', () => {
    mainWindow.minimize();
});

ipcMain.handle('close-window', () => {
    mainWindow.close();
});

ipcMain.handle('drag-window', (event, { x, y }) => {
    const [currentX, currentY] = mainWindow.getPosition();
    mainWindow.setPosition(currentX + x, currentY + y);
});

ipcMain.handle('set-ignore-mouse-events', (event, ignore, options) => {
    mainWindow.setIgnoreMouseEvents(ignore, options);
});
