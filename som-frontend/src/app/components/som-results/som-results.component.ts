import { Component } from '@angular/core';
import { SomService } from '../../services/som.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-som-results',
  standalone: true,
  templateUrl: './som-results.component.html',
  styleUrls: ['./som-results.component.scss'],
  imports: [CommonModule]
})
export class SomResultsComponent {
  imageUrl: string | undefined;
  isLoading: boolean = false; // Variável para rastrear o estado de carregamento

  constructor(private somService: SomService) { }

  trainSom() {
    console.log("Treinando SOM...");
    this.isLoading = true; // Inicia o carregamento
    this.somService.trainSom().subscribe({
      next: (response) => {
        console.log('Treinamento concluído', response);
        this.fetchImage();
      },
      error: (err) => {
        console.error('Erro ao treinar SOM:', err);
      },
      complete: () => {
        this.isLoading = false; // Termina o carregamento
      }
    });
  }

  fetchImage() {
    this.somService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrl = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem:', err);
      }
    });
  }
}