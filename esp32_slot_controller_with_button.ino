#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ESP32Servo.h>

// サーボモーター設定
Servo myservo;
int servoPin = 18;

// ボタン設定
const int buttonPin = 2;  // GPIO2にプルアップ抵抗付きボタン接続
bool servoEnabled = true; // サーボ動作有効/無効フラグ
bool lastButtonState = HIGH;
bool buttonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// BLE設定
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;

#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("BLE接続されました");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("BLE切断されました");
    }
};

class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String value = pCharacteristic->getValue();
      
      if (value.length() > 0) {
        Serial.println("受信したコマンド: " + value);
        
        if (value == "PRESS") {
          if (servoEnabled) {
            pressButton();
            pCharacteristic->setValue("PRESS_COMPLETE");
          } else {
            Serial.println("サーボ無効状態のため動作をスキップします");
            pCharacteristic->setValue("SERVO_DISABLED");
          }
          pCharacteristic->notify();
        }
      }
    }
};

void setup() {
  Serial.begin(115200);
  Serial.println("スロット制御システム初期化開始...");
  
  // ピン設定
  pinMode(buttonPin, INPUT_PULLUP);  // 内蔵プルアップ抵抗使用
  
  // サーボモーター初期化
  myservo.attach(servoPin);
  myservo.write(0);  // 初期位置
  delay(500);
  
  // BLE初期化
  BLEDevice::init("ESP32SlotController");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_WRITE |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->setCallbacks(new MyCallbacks());
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();
  
  Serial.println("BLEサーバー開始完了");
  Serial.println("サーボ制御: " + String(servoEnabled ? "有効" : "無効"));
  Serial.println("ボタン操作でサーボON/OFF切替可能");
}

void loop() {
  // ボタン状態チェック
  checkButton();
  
  // BLE接続状態の監視
  if (!deviceConnected && pServer->getConnectedCount() == 0) {
    delay(500);
    pServer->startAdvertising();
  }
  
  delay(10); // CPU負荷軽減
}

void checkButton() {
  int reading = digitalRead(buttonPin);
  
  // デバウンス処理
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      
      // ボタンが押された時（HIGH→LOW）
      if (buttonState == LOW) {
        servoEnabled = !servoEnabled;
        
        Serial.println("ボタン押下: サーボ制御 " + String(servoEnabled ? "有効" : "無効"));
        
        // BLE経由で状態通知
        if (deviceConnected) {
          String status = servoEnabled ? "SERVO_ENABLED" : "SERVO_DISABLED";
          pCharacteristic->setValue(status.c_str());
          pCharacteristic->notify();
        }
      }
    }
  }
  
  lastButtonState = reading;
}


void pressButton() {
  Serial.println("ボタンプレス実行開始");
  
  // サーボモーター動作
  myservo.write(90);   // ボタンを押す位置
  delay(100);          // 押し続ける時間
  myservo.write(0);    // 元の位置に戻る
  delay(100);          // 安定待ち
  
  Serial.println("ボタンプレス完了");
}