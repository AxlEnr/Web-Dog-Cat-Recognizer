import { Component, OnInit, Inject, PLATFORM_ID, NgModule } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import * as tf from '@tensorflow/tfjs';
import { FormsModule } from '@angular/forms'; 
@Component({
  selector: 'app-root',
  standalone: true,
  imports:[CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  // Estado de la Carga
  loading = true;
  modelsLoaded = false;
  predicting = false;
  error?: string;

  // Modelos de IA
  private resnetModel?: tf.GraphModel;
  private vgg16Model?: tf.GraphModel;

  // Datos de la Interfaz
  selectedModel: 'resnet' | 'vgg16' = 'vgg16';
  imagePreviewUrl: string | ArrayBuffer | null = null;
  imageFile: File | null = null;
  prediction: 'Perro' | 'Gato' | null = null;

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {}

  ngOnInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.loadModels();
    } else {
      console.log('Omitiendo la carga de modelos en el SERVIDOR.');
      this.loading = false;
    }
  }

  async loadModels() {
    try {
      this.loading = true;
      console.log('Cargando modelos...');
      [this.resnetModel, this.vgg16Model] = await Promise.all([
        tf.loadGraphModel('/assets/resnet_tfjs_model_dir/model.json'),
        tf.loadGraphModel('/assets/vgg16_tfjs_model_dir/model.json')
      ]);
      console.log('Modelos cargados exitosamente.');
      this.modelsLoaded = true;
    } catch (err) {
      this.error = 'Falló la carga de los modelos.';
      console.error(err);
    } finally {
      this.loading = false;
    }
  }

  onFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      this.imageFile = target.files[0];
      this.prediction = null; 

      // Crear vista previa de la imagen
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreviewUrl = reader.result;
      };
      reader.readAsDataURL(this.imageFile);
    }
  }

  async startPrediction(): Promise<void> {
    if (!this.imageFile || (!this.resnetModel || !this.vgg16Model)) {
      console.error('Archivo o modelo no disponible.');
      return;
    }

    this.predicting = true;
    this.prediction = null;

    // Crear un elemento de imagen para que TFJS pueda procesarlo
    const imageElement = document.createElement('img');
    imageElement.src = this.imagePreviewUrl as string;

    imageElement.onload = async () => {
      // tf.tidy() limpia la memoria de los tensores automáticamente
      const result = tf.tidy(() => {
        // 1. Convertir la imagen a un tensor
        const imgTensor = tf.browser.fromPixels(imageElement).toFloat();

        // 2. Pre-procesamiento: Redimensionar y normalizar
        // Los modelos como ResNet/VGG16 esperan un tamaño específico (ej. 224x224)
        // La normalización depende de cómo fue entrenado el modelo.
        // Una normalización común es escalar los pixeles de 0-255 a 0-1.
        const resized = tf.image.resizeBilinear(imgTensor, [224, 224]);
        const normalized = resized.div(255.0);

        // 3. Añadir una dimensión de "batch" (lote)
        // La forma pasa de [224, 224, 3] a [1, 224, 224, 3]
        const batched = normalized.expandDims(0);
        
        // 4. Seleccionar el modelo y predecir
        const modelToUse = this.selectedModel === 'resnet' ? this.resnetModel : this.vgg16Model;
        return modelToUse?.predict(batched) as tf.Tensor;
      });

      // 5. Interpretar el resultado
      const predictionData = await result.data();
      result.dispose(); // Liberar memoria del tensor de resultado

      // Suponemos que el modelo devuelve un solo número.
      // < 0.5 = Gato, > 0.5 = Perro (esto puede variar según tu modelo)
      this.prediction = predictionData[0] > 0.5 ? 'Perro' : 'Gato';
      
      this.predicting = false;
    };
  }
}