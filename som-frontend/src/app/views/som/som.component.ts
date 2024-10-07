import { Component, ViewChild } from '@angular/core';
import { NavComponent } from '../../components/nav/nav.component';
import { ButtonComponent } from '../../components/button/button.component';
import { SomResultsComponent } from '../../components/som-results/som-results.component';

@Component({
  selector: 'app-som',
  standalone: true,
  imports: [NavComponent, ButtonComponent, SomResultsComponent],
  templateUrl: './som.component.html',
  styleUrls: ['./som.component.scss']
})
export class SomComponent {
  @ViewChild(SomResultsComponent) somResultsComponent!: SomResultsComponent;

  onTrainSom() {
    if (this.somResultsComponent) {
      this.somResultsComponent.trainSom();
    }
  }
}
