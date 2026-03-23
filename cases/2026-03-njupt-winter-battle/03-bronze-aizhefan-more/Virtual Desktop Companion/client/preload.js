const { ipcRenderer, contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    minimize: () => ipcRenderer.invoke('minimize-window'),
    close: () => ipcRenderer.invoke('close-window'),
    drag: (x, y) => ipcRenderer.invoke('drag-window', { x, y }),
    setIgnoreMouseEvents: (ignore, options) => ipcRenderer.invoke('set-ignore-mouse-events', ignore, options)
});
