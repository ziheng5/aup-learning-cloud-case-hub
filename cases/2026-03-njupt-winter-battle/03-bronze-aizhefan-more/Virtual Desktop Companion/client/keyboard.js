const { ipcMain } = require('electron');
const { exec } = require('child_process');

let robot;
try {
    robot = require('robotjs');
} catch (e) {
    console.log('robotjs not available, keyboard control disabled');
}

function executeKeyboardCommand(command) {
    if (!robot) {
        console.log('Keyboard control not available');
        return;
    }
    
    const { type, key, modifiers, text, delay } = command;
    
    switch (type) {
        case 'type':
            if (text) {
                robot.typeString(text);
            }
            break;
            
        case 'tap':
            if (key) {
                if (modifiers && modifiers.length > 0) {
                    robot.keyTap(key, modifiers);
                } else {
                    robot.keyTap(key);
                }
            }
            break;
            
        case 'press':
            if (key) {
                robot.keyToggle(key, 'down');
                setTimeout(() => {
                    robot.keyToggle(key, 'up');
                }, delay || 100);
            }
            break;
            
        case 'shortcut':
            if (modifiers && modifiers.length > 0 && key) {
                robot.keyTap(key, modifiers);
            }
            break;
            
        case 'sequence':
            if (Array.isArray(key)) {
                key.forEach((k, i) => {
                    setTimeout(() => {
                        robot.keyTap(k);
                    }, i * (delay || 100));
                });
            }
            break;
    }
}

ipcMain.handle('execute-keyboard', async (event, command) => {
    try {
        executeKeyboardCommand(command);
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

module.exports = { executeKeyboardCommand };
