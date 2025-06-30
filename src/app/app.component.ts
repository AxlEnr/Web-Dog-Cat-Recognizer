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
  loading = true;
  modelsLoaded = false;
  predicting = false;
  error?: string;

  private resnetModel?: tf.GraphModel;
  private vgg16Model?: tf.GraphModel;
  private mobilenetModel?: tf.GraphModel;

  selectedModel: 'resnet' | 'mobilenet' | 'vgg16' = 'vgg16';
  imagePreviewUrl: string | ArrayBuffer | null = null;
  imageFile: File | null = null;
  prediction: 'Perro' | 'Gato' | null = null;

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {}

  ngOnInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.loadModels();
    } else {
      this.loading = false;
    }
  }

  async loadModels() {
    try {
      this.loading = true;
      console.log('Cargando modelos...');
      [this.resnetModel, this.vgg16Model, this.mobilenetModel] = await Promise.all([
        tf.loadGraphModel('/assets/resnet_tfjs_model_dir/model.json'),
        tf.loadGraphModel('/assets/vgg16_tfjs_model_dir/model.json'),
        tf.loadGraphModel('/assets/mobilenet_tfjs_model_dir/model.json')
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

      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreviewUrl = reader.result;
      };
      reader.readAsDataURL(this.imageFile);
    }
  }

  async startPrediction(): Promise<void> {
    if (!this.imageFile || (!this.resnetModel || !this.vgg16Model || !this.mobilenetModel)) {
      console.error('Archivo o modelo no disponible.');
      return;
    }

    this.predicting = true;
    this.prediction = null;

    const imageElement = document.createElement('img');
    imageElement.src = this.imagePreviewUrl as string;

    imageElement.onload = async () => {
      const result = tf.tidy(() => {

        const imgTensor = tf.browser.fromPixels(imageElement).toFloat();

        const resized = tf.image.resizeBilinear(imgTensor, [224, 224]);
        const normalized = resized.div(255.0);

        const batched = normalized.expandDims(0);

        const modelToUse = this.selectedModel === 'resnet'
          ? this.resnetModel
          : this.selectedModel === 'vgg16'
            ? this.vgg16Model
            : this.mobilenetModel;
        return modelToUse?.predict(batched) as tf.Tensor;
      });


      const predictionData = await result.data();
      result.dispose();

      this.prediction = predictionData[0] > 0.5 ? 'Gato' : 'Perro';
      
      this.predicting = false;
    };
  }
}